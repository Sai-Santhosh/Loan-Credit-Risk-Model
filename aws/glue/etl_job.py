"""
AWS Glue ETL Job - Credit Risk Data Processing

Production ETL pipeline for processing loan data.
Runs as a Glue Spark job for large-scale data processing.
"""

import sys
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, IntegerType

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'source_database',
    'source_table',
    'target_bucket',
    'target_prefix'
])

job.init(args['JOB_NAME'], args)

# Configuration
SOURCE_DATABASE = args['source_database']
SOURCE_TABLE = args['source_table']
TARGET_BUCKET = args['target_bucket']
TARGET_PREFIX = args['target_prefix']

print(f"Starting ETL Job: {args['JOB_NAME']}")
print(f"Source: {SOURCE_DATABASE}.{SOURCE_TABLE}")
print(f"Target: s3://{TARGET_BUCKET}/{TARGET_PREFIX}")


def load_source_data():
    """Load data from Glue Data Catalog."""
    print("Loading source data...")
    
    dyf = glueContext.create_dynamic_frame.from_catalog(
        database=SOURCE_DATABASE,
        table_name=SOURCE_TABLE,
        transformation_ctx="source_data"
    )
    
    print(f"Loaded {dyf.count()} records")
    return dyf


def clean_data(dyf):
    """Clean and filter data."""
    print("Cleaning data...")
    
    # Convert to Spark DataFrame for transformations
    df = dyf.toDF()
    
    # Drop unnecessary columns
    drop_columns = [
        'Unnamed: 0', 'emp_title', 'title', 'address',
        'issue_d', 'sub_grade', 'application_type'
    ]
    
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(col)
    
    # Filter valid loan status
    df = df.filter(F.col('loan_status').isin(['Fully Paid', 'Charged Off']))
    
    # Filter positive annual income
    df = df.filter(F.col('annual_inc') > 0)
    
    print(f"After cleaning: {df.count()} records")
    return df


def handle_missing_values(df):
    """Handle missing values."""
    print("Handling missing values...")
    
    # Calculate mort_acc medians by total_acc
    mort_acc_medians = df.groupBy('total_acc').agg(
        F.expr('percentile_approx(mort_acc, 0.5)').alias('median_mort_acc')
    )
    
    # Join and fill mort_acc
    df = df.join(mort_acc_medians, on='total_acc', how='left')
    df = df.withColumn(
        'mort_acc',
        F.coalesce(F.col('mort_acc'), F.col('median_mort_acc'))
    ).drop('median_mort_acc')
    
    # Fill remaining numerical nulls with median
    numeric_cols = ['revol_util', 'pub_rec_bankruptcies']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df.approxQuantile(col, [0.5], 0.01)[0]
            df = df.fillna({col: median_val})
    
    # Fill emp_length nulls
    df = df.fillna({'emp_length': 'Unknown'})
    
    return df


def engineer_features(df):
    """Create engineered features."""
    print("Engineering features...")
    
    # Loan to income ratio
    df = df.withColumn(
        'loan_to_income',
        F.col('loan_amnt') / F.col('annual_inc')
    )
    
    # Total interest owed
    df = df.withColumn(
        'total_interest_owed',
        F.col('loan_amnt') * (F.col('int_rate') / 100)
    )
    
    # Installment to income ratio
    df = df.withColumn(
        'installment_to_income_ratio',
        F.col('installment') / (F.col('annual_inc') / 12)
    )
    
    # Active credit percentage
    df = df.withColumn(
        'active_credit_pct',
        F.when(F.col('total_acc') > 0,
               F.col('open_acc') / F.col('total_acc'))
        .otherwise(None)
    )
    
    # Credit age (years since earliest credit line)
    current_year = datetime.now().year
    df = df.withColumn(
        'credit_age',
        F.lit(current_year) - F.year(F.to_date(F.col('earliest_cr_line'), 'MMM-yyyy'))
    )
    
    # Convert emp_length to numeric
    df = df.withColumn(
        'emp_length_numeric',
        F.when(F.col('emp_length') == '10+ years', 10)
        .when(F.col('emp_length') == '< 1 year', 0)
        .when(F.col('emp_length') == 'Unknown', -1)
        .otherwise(F.regexp_extract(F.col('emp_length'), r'(\d+)', 1).cast(IntegerType()))
    )
    
    # Bin public records
    df = df.withColumn(
        'pub_rec_binned',
        F.when(F.col('pub_rec') == 0, '0')
        .when(F.col('pub_rec') == 1, '1')
        .otherwise('2+')
    )
    
    # Bin pub_rec_bankruptcies
    df = df.withColumn(
        'pub_rec_bankruptcies_binned',
        F.when(F.col('pub_rec_bankruptcies') == 0, '0')
        .when(F.col('pub_rec_bankruptcies') == 1, '1')
        .otherwise('2+')
    )
    
    return df


def select_final_features(df):
    """Select final features for model training."""
    print("Selecting final features...")
    
    feature_columns = [
        'term', 'int_rate', 'grade', 'emp_length_numeric',
        'home_ownership', 'annual_inc', 'verification_status',
        'purpose', 'dti', 'pub_rec_binned', 'revol_util',
        'initial_list_status', 'mort_acc', 'pub_rec_bankruptcies_binned',
        'loan_to_income', 'total_interest_owed',
        'installment_to_income_ratio', 'active_credit_pct', 'credit_age',
        'loan_status'
    ]
    
    # Only select columns that exist
    available_columns = [col for col in feature_columns if col in df.columns]
    df = df.select(available_columns)
    
    # Rename columns for consistency
    df = df.withColumnRenamed('emp_length_numeric', 'emp_length')
    df = df.withColumnRenamed('pub_rec_binned', 'pub_rec')
    df = df.withColumnRenamed('pub_rec_bankruptcies_binned', 'pub_rec_bankruptcies')
    
    # Drop rows with remaining nulls
    df = df.dropna()
    
    print(f"Final dataset: {df.count()} records, {len(df.columns)} columns")
    return df


def add_metadata(df):
    """Add processing metadata."""
    df = df.withColumn('etl_timestamp', F.current_timestamp())
    df = df.withColumn('etl_job_name', F.lit(args['JOB_NAME']))
    return df


def write_output(df):
    """Write processed data to S3."""
    print("Writing output...")
    
    output_path = f"s3://{TARGET_BUCKET}/{TARGET_PREFIX}"
    
    # Write as Parquet with partitioning
    df.write \
        .mode('overwrite') \
        .partitionBy('loan_status') \
        .parquet(output_path)
    
    print(f"Output written to {output_path}")
    
    # Also write as single file for easy access
    single_file_path = f"s3://{TARGET_BUCKET}/{TARGET_PREFIX}_combined"
    df.coalesce(1).write \
        .mode('overwrite') \
        .parquet(single_file_path)
    
    print(f"Combined file written to {single_file_path}")


def generate_statistics(df):
    """Generate and log data statistics."""
    print("\n" + "=" * 50)
    print("DATA STATISTICS")
    print("=" * 50)
    
    # Row count
    total_rows = df.count()
    print(f"Total rows: {total_rows}")
    
    # Target distribution
    target_dist = df.groupBy('loan_status').count().collect()
    for row in target_dist:
        pct = (row['count'] / total_rows) * 100
        print(f"  {row['loan_status']}: {row['count']} ({pct:.2f}%)")
    
    # Feature statistics
    print("\nNumerical Feature Statistics:")
    numeric_cols = ['int_rate', 'annual_inc', 'dti', 'revol_util']
    for col in numeric_cols:
        if col in df.columns:
            stats = df.select(
                F.mean(col).alias('mean'),
                F.stddev(col).alias('std'),
                F.min(col).alias('min'),
                F.max(col).alias('max')
            ).collect()[0]
            print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    print("=" * 50 + "\n")


def main():
    """Main ETL workflow."""
    try:
        # Load data
        source_dyf = load_source_data()
        
        # Clean data
        df = clean_data(source_dyf)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Engineer features
        df = engineer_features(df)
        
        # Select final features
        df = select_final_features(df)
        
        # Add metadata
        df = add_metadata(df)
        
        # Generate statistics
        generate_statistics(df)
        
        # Write output
        write_output(df)
        
        print("ETL Job completed successfully!")
        
    except Exception as e:
        print(f"ETL Job failed: {str(e)}")
        raise


# Run main
main()

# Commit job
job.commit()
