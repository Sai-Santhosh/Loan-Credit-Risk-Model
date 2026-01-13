"""
AWS Lambda Handler - Data Ingestion

Handles data ingestion from various sources to S3.
Triggered by S3 events, API Gateway, or scheduled CloudWatch events.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict
import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client("s3")
glue_client = boto3.client("glue")

# Environment variables
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "credit-risk-ml-pipeline")
RAW_DATA_PREFIX = os.environ.get("RAW_DATA_PREFIX", "data/raw/")
GLUE_CRAWLER_NAME = os.environ.get("GLUE_CRAWLER_NAME", "credit-risk-crawler")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for data ingestion.
    
    Supports:
    - S3 event triggers (new file upload)
    - API Gateway requests (manual trigger)
    - CloudWatch scheduled events
    
    Args:
        event: Lambda event object
        context: Lambda context object
    
    Returns:
        Response dictionary
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Determine event source
        if "Records" in event:
            # S3 event trigger
            return handle_s3_event(event)
        elif "httpMethod" in event:
            # API Gateway request
            return handle_api_request(event)
        elif "source" in event and event["source"] == "aws.events":
            # CloudWatch scheduled event
            return handle_scheduled_event(event)
        else:
            return handle_direct_invocation(event)
    
    except Exception as e:
        logger.error(f"Error processing event: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        }


def handle_s3_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle S3 event trigger when new data file is uploaded."""
    processed_files = []
    
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        
        logger.info(f"Processing file: s3://{bucket}/{key}")
        
        # Validate file
        if not is_valid_data_file(key):
            logger.warning(f"Skipping invalid file: {key}")
            continue
        
        # Log file metadata
        response = s3_client.head_object(Bucket=bucket, Key=key)
        file_size = response["ContentLength"]
        
        logger.info(f"File size: {file_size} bytes")
        
        # Record ingestion metadata
        metadata = {
            "source_bucket": bucket,
            "source_key": key,
            "file_size_bytes": file_size,
            "ingestion_timestamp": datetime.now().isoformat(),
            "status": "ingested"
        }
        
        # Save metadata
        save_ingestion_metadata(key, metadata)
        
        processed_files.append({
            "bucket": bucket,
            "key": key,
            "size": file_size
        })
    
    # Trigger Glue Crawler to update catalog
    if processed_files:
        trigger_glue_crawler()
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Data ingestion completed",
            "processed_files": processed_files,
            "timestamp": datetime.now().isoformat()
        })
    }


def handle_api_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle API Gateway request for manual data ingestion."""
    body = json.loads(event.get("body", "{}"))
    
    source_path = body.get("source_path")
    destination_key = body.get("destination_key")
    
    if not source_path:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "source_path is required"})
        }
    
    logger.info(f"Manual ingestion requested for: {source_path}")
    
    # Copy data to S3
    if destination_key is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_key = f"{RAW_DATA_PREFIX}manual_upload_{timestamp}.csv"
    
    # Here you would implement the actual data transfer logic
    # For example, fetching from an external API or database
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "message": "Manual ingestion initiated",
            "destination": f"s3://{BUCKET_NAME}/{destination_key}",
            "timestamp": datetime.now().isoformat()
        })
    }


def handle_scheduled_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle scheduled CloudWatch event for periodic ingestion."""
    logger.info("Processing scheduled data ingestion")
    
    # List recent files in raw data prefix
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=RAW_DATA_PREFIX,
        MaxKeys=100
    )
    
    file_count = response.get("KeyCount", 0)
    logger.info(f"Found {file_count} files in raw data prefix")
    
    # Trigger Glue Crawler
    trigger_glue_crawler()
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Scheduled ingestion completed",
            "files_found": file_count,
            "timestamp": datetime.now().isoformat()
        })
    }


def handle_direct_invocation(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle direct Lambda invocation."""
    action = event.get("action", "status")
    
    if action == "status":
        return get_ingestion_status()
    elif action == "trigger_crawler":
        trigger_glue_crawler()
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Crawler triggered"})
        }
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Unknown action: {action}"})
        }


def is_valid_data_file(key: str) -> bool:
    """Validate if the file is a valid data file."""
    valid_extensions = [".csv", ".parquet", ".json"]
    return any(key.lower().endswith(ext) for ext in valid_extensions)


def save_ingestion_metadata(key: str, metadata: Dict[str, Any]) -> None:
    """Save ingestion metadata to S3."""
    metadata_key = f"metadata/ingestion/{key.replace('/', '_')}.json"
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=metadata_key,
        Body=json.dumps(metadata),
        ContentType="application/json"
    )
    
    logger.info(f"Saved metadata to {metadata_key}")


def trigger_glue_crawler() -> None:
    """Trigger AWS Glue Crawler to update data catalog."""
    try:
        glue_client.start_crawler(Name=GLUE_CRAWLER_NAME)
        logger.info(f"Triggered Glue Crawler: {GLUE_CRAWLER_NAME}")
    except glue_client.exceptions.CrawlerRunningException:
        logger.info("Crawler already running")
    except Exception as e:
        logger.error(f"Failed to trigger crawler: {e}")


def get_ingestion_status() -> Dict[str, Any]:
    """Get current ingestion status."""
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=RAW_DATA_PREFIX
    )
    
    files = []
    total_size = 0
    
    for obj in response.get("Contents", []):
        files.append({
            "key": obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"].isoformat()
        })
        total_size += obj["Size"]
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "bucket": BUCKET_NAME,
            "prefix": RAW_DATA_PREFIX,
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "recent_files": files[:10],
            "timestamp": datetime.now().isoformat()
        })
    }
