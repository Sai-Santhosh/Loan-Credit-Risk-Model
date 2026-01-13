"""
Data Loader Module - Production-grade data loading utilities.

Supports loading data from:
- Local filesystem (CSV, Parquet)
- AWS S3
- AWS Redshift
- AWS Athena
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def save(self, df: pd.DataFrame, destination: str, **kwargs) -> None:
        """Save data to destination."""
        pass


class DataLoader(BaseDataLoader):
    """
    Local filesystem data loader.
    
    Supports CSV, Parquet, and other common formats.
    
    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("data/raw/train.csv")
        >>> loader.save(df, "data/processed/train.parquet")
    """
    
    SUPPORTED_FORMATS = {".csv", ".parquet", ".json", ".xlsx"}
    
    def __init__(self, default_format: str = "parquet"):
        """
        Initialize DataLoader.
        
        Args:
            default_format: Default format for saving (parquet, csv)
        """
        self.default_format = default_format
    
    def load(
        self,
        source: str,
        format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from local filesystem.
        
        Args:
            source: Path to data file
            format: File format (auto-detected if None)
            **kwargs: Additional arguments passed to pandas reader
        
        Returns:
            Loaded DataFrame
        """
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {source}")
        
        file_format = format or path.suffix.lower()
        
        logger.info(f"Loading data from {source} (format: {file_format})")
        
        if file_format == ".csv":
            df = pd.read_csv(source, **kwargs)
        elif file_format == ".parquet":
            df = pd.read_parquet(source, **kwargs)
        elif file_format == ".json":
            df = pd.read_json(source, **kwargs)
        elif file_format == ".xlsx":
            df = pd.read_excel(source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def save(
        self,
        df: pd.DataFrame,
        destination: str,
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save DataFrame to local filesystem.
        
        Args:
            df: DataFrame to save
            destination: Output path
            format: File format (auto-detected if None)
            **kwargs: Additional arguments passed to pandas writer
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = format or path.suffix.lower() or f".{self.default_format}"
        
        logger.info(f"Saving {len(df)} rows to {destination}")
        
        if file_format == ".csv":
            df.to_csv(destination, index=False, **kwargs)
        elif file_format == ".parquet":
            df.to_parquet(destination, index=False, **kwargs)
        elif file_format == ".json":
            df.to_json(destination, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Successfully saved to {destination}")


class S3DataLoader(BaseDataLoader):
    """
    AWS S3 data loader using boto3 and awswrangler.
    
    Example:
        >>> loader = S3DataLoader(bucket="my-bucket")
        >>> df = loader.load("data/raw/train.parquet")
        >>> loader.save(df, "data/processed/train.parquet")
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize S3DataLoader.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            aws_access_key_id: Optional AWS access key
            aws_secret_access_key: Optional AWS secret key
        """
        self.bucket = bucket
        self.region = region
        
        # Import AWS libraries
        try:
            import boto3
            import awswrangler as wr
            self.boto3 = boto3
            self.wr = wr
        except ImportError:
            raise ImportError(
                "AWS dependencies not installed. "
                "Install with: pip install boto3 awswrangler"
            )
        
        # Configure boto3 session
        session_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        
        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client("s3")
    
    def _get_s3_path(self, key: str) -> str:
        """Construct full S3 path."""
        return f"s3://{self.bucket}/{key}"
    
    def load(
        self,
        source: str,
        format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from S3.
        
        Args:
            source: S3 key (path within bucket)
            format: File format (auto-detected if None)
            **kwargs: Additional arguments
        
        Returns:
            Loaded DataFrame
        """
        s3_path = self._get_s3_path(source)
        file_format = format or Path(source).suffix.lower()
        
        logger.info(f"Loading data from {s3_path}")
        
        if file_format == ".parquet":
            df = self.wr.s3.read_parquet(s3_path, boto3_session=self.session, **kwargs)
        elif file_format == ".csv":
            df = self.wr.s3.read_csv(s3_path, boto3_session=self.session, **kwargs)
        elif file_format == ".json":
            df = self.wr.s3.read_json(s3_path, boto3_session=self.session, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Loaded {len(df)} rows from S3")
        return df
    
    def save(
        self,
        df: pd.DataFrame,
        destination: str,
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save DataFrame to S3.
        
        Args:
            df: DataFrame to save
            destination: S3 key (path within bucket)
            format: File format
            **kwargs: Additional arguments
        """
        s3_path = self._get_s3_path(destination)
        file_format = format or Path(destination).suffix.lower()
        
        logger.info(f"Saving {len(df)} rows to {s3_path}")
        
        if file_format == ".parquet":
            self.wr.s3.to_parquet(df, s3_path, boto3_session=self.session, **kwargs)
        elif file_format == ".csv":
            self.wr.s3.to_csv(df, s3_path, index=False, boto3_session=self.session, **kwargs)
        elif file_format == ".json":
            self.wr.s3.to_json(df, s3_path, boto3_session=self.session, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        logger.info(f"Successfully saved to {s3_path}")
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 bucket with given prefix."""
        objects = self.wr.s3.list_objects(
            f"s3://{self.bucket}/{prefix}",
            boto3_session=self.session
        )
        return objects


class RedshiftDataLoader(BaseDataLoader):
    """
    AWS Redshift data loader.
    
    Example:
        >>> loader = RedshiftDataLoader(
        ...     host="cluster.redshift.amazonaws.com",
        ...     database="mydb",
        ...     user="admin",
        ...     password="secret"
        ... )
        >>> df = loader.load("SELECT * FROM credit_data")
    """
    
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5439,
        schema: str = "public",
    ):
        """
        Initialize RedshiftDataLoader.
        
        Args:
            host: Redshift cluster endpoint
            database: Database name
            user: Username
            password: Password
            port: Port number
            schema: Default schema
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.schema = schema
        
        try:
            import awswrangler as wr
            self.wr = wr
        except ImportError:
            raise ImportError(
                "awswrangler not installed. Install with: pip install awswrangler"
            )
        
        self.conn_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from Redshift using SQL query.
        
        Args:
            source: SQL query string
            **kwargs: Additional arguments
        
        Returns:
            Query results as DataFrame
        """
        logger.info(f"Executing Redshift query...")
        
        with self.wr.redshift.connect(**self.conn_params) as conn:
            df = self.wr.redshift.read_sql_query(source, con=conn, **kwargs)
        
        logger.info(f"Loaded {len(df)} rows from Redshift")
        return df
    
    def save(
        self,
        df: pd.DataFrame,
        destination: str,
        mode: str = "append",
        **kwargs
    ) -> None:
        """
        Save DataFrame to Redshift table.
        
        Args:
            df: DataFrame to save
            destination: Table name
            mode: Write mode (append, overwrite)
            **kwargs: Additional arguments
        """
        logger.info(f"Writing {len(df)} rows to Redshift table {destination}")
        
        with self.wr.redshift.connect(**self.conn_params) as conn:
            self.wr.redshift.to_sql(
                df=df,
                con=conn,
                table=destination,
                schema=self.schema,
                mode=mode,
                **kwargs
            )
        
        logger.info(f"Successfully wrote to {destination}")
    
    def execute(self, query: str) -> None:
        """Execute a SQL statement."""
        with self.wr.redshift.connect(**self.conn_params) as conn:
            conn.execute(query)


class AthenaDataLoader(BaseDataLoader):
    """
    AWS Athena data loader for querying data lake.
    
    Example:
        >>> loader = AthenaDataLoader(
        ...     database="credit_risk_db",
        ...     output_location="s3://my-bucket/athena-results/"
        ... )
        >>> df = loader.load("SELECT * FROM loan_data WHERE year = 2024")
    """
    
    def __init__(
        self,
        database: str,
        output_location: str,
        workgroup: str = "primary",
        region: str = "us-east-1",
    ):
        """
        Initialize AthenaDataLoader.
        
        Args:
            database: Athena database name
            output_location: S3 path for query results
            workgroup: Athena workgroup
            region: AWS region
        """
        self.database = database
        self.output_location = output_location
        self.workgroup = workgroup
        self.region = region
        
        try:
            import awswrangler as wr
            self.wr = wr
        except ImportError:
            raise ImportError(
                "awswrangler not installed. Install with: pip install awswrangler"
            )
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Execute Athena query and return results.
        
        Args:
            source: SQL query string
            **kwargs: Additional arguments
        
        Returns:
            Query results as DataFrame
        """
        logger.info(f"Executing Athena query...")
        
        df = self.wr.athena.read_sql_query(
            sql=source,
            database=self.database,
            s3_output=self.output_location,
            workgroup=self.workgroup,
            **kwargs
        )
        
        logger.info(f"Loaded {len(df)} rows from Athena")
        return df
    
    def save(self, df: pd.DataFrame, destination: str, **kwargs) -> None:
        """
        Save DataFrame to S3 and register as Athena table.
        
        Args:
            df: DataFrame to save
            destination: Table name
            **kwargs: Additional arguments
        """
        s3_path = f"{self.output_location.rstrip('/')}/tables/{destination}/"
        
        logger.info(f"Writing {len(df)} rows to Athena table {destination}")
        
        self.wr.s3.to_parquet(
            df=df,
            path=s3_path,
            dataset=True,
            database=self.database,
            table=destination,
            **kwargs
        )
        
        logger.info(f"Successfully wrote to {destination}")


__all__ = [
    "DataLoader",
    "S3DataLoader",
    "RedshiftDataLoader",
    "AthenaDataLoader",
]
