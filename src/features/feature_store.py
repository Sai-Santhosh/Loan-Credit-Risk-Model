"""
Feature Store Module - Offline and online feature serving.

Provides a simple feature store abstraction for:
- Storing computed features
- Serving features for training and inference
- Feature versioning and metadata
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import json
import hashlib

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Simple feature store for credit risk features.
    
    Supports:
    - Local file-based storage (Parquet)
    - S3 storage (via S3DataLoader)
    - Feature metadata tracking
    - Feature versioning
    
    Example:
        >>> store = FeatureStore(base_path="data/feature_store")
        >>> store.save_features(df, "training_features", version="1.0.0")
        >>> features = store.load_features("training_features")
    """
    
    def __init__(
        self,
        base_path: str = "data/feature_store",
        s3_bucket: Optional[str] = None,
    ):
        """
        Initialize FeatureStore.
        
        Args:
            base_path: Local base path for feature storage
            s3_bucket: Optional S3 bucket for cloud storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        # Load or create feature registry
        self.registry_path = self.metadata_path / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load feature registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"features": {}}
    
    def _save_registry(self) -> None:
        """Save feature registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for versioning."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:12]
    
    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        version: Optional[str] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save features to store.
        
        Args:
            df: DataFrame with features
            name: Feature set name
            version: Version string (auto-generated if None)
            description: Description of feature set
            tags: Additional metadata tags
        
        Returns:
            Feature set ID
        """
        # Auto-generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        feature_id = f"{name}_v{version}"
        feature_path = self.base_path / name / version
        feature_path.mkdir(parents=True, exist_ok=True)
        
        # Save features as parquet
        data_path = feature_path / "features.parquet"
        df.to_parquet(data_path, index=False)
        
        # Compute metadata
        metadata = {
            "id": feature_id,
            "name": name,
            "version": version,
            "description": description,
            "tags": tags or {},
            "created_at": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "data_hash": self._compute_hash(df),
            "storage_path": str(data_path),
        }
        
        # Save metadata
        meta_path = feature_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        if name not in self.registry["features"]:
            self.registry["features"][name] = {"versions": []}
        
        self.registry["features"][name]["versions"].append({
            "version": version,
            "created_at": metadata["created_at"],
            "row_count": metadata["row_count"],
        })
        self.registry["features"][name]["latest"] = version
        self._save_registry()
        
        logger.info(f"Saved feature set: {feature_id} ({len(df)} rows)")
        
        return feature_id
    
    def load_features(
        self,
        name: str,
        version: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load features from store.
        
        Args:
            name: Feature set name
            version: Version to load (latest if None)
            columns: Specific columns to load
        
        Returns:
            DataFrame with features
        """
        # Get version
        if version is None:
            if name not in self.registry["features"]:
                raise ValueError(f"Feature set not found: {name}")
            version = self.registry["features"][name]["latest"]
        
        feature_path = self.base_path / name / version / "features.parquet"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Features not found: {name} v{version}")
        
        df = pd.read_parquet(feature_path, columns=columns)
        
        logger.info(f"Loaded feature set: {name} v{version} ({len(df)} rows)")
        
        return df
    
    def get_metadata(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for feature set.
        
        Args:
            name: Feature set name
            version: Version (latest if None)
        
        Returns:
            Metadata dictionary
        """
        if version is None:
            version = self.registry["features"][name]["latest"]
        
        meta_path = self.base_path / name / version / "metadata.json"
        
        with open(meta_path, "r") as f:
            return json.load(f)
    
    def list_feature_sets(self) -> pd.DataFrame:
        """List all feature sets in store."""
        records = []
        
        for name, info in self.registry["features"].items():
            records.append({
                "name": name,
                "latest_version": info.get("latest"),
                "num_versions": len(info.get("versions", [])),
            })
        
        return pd.DataFrame(records)
    
    def list_versions(self, name: str) -> pd.DataFrame:
        """List all versions of a feature set."""
        if name not in self.registry["features"]:
            raise ValueError(f"Feature set not found: {name}")
        
        return pd.DataFrame(self.registry["features"][name]["versions"])
    
    def delete_version(self, name: str, version: str) -> None:
        """Delete a specific version of features."""
        import shutil
        
        feature_path = self.base_path / name / version
        
        if feature_path.exists():
            shutil.rmtree(feature_path)
        
        # Update registry
        if name in self.registry["features"]:
            self.registry["features"][name]["versions"] = [
                v for v in self.registry["features"][name]["versions"]
                if v["version"] != version
            ]
            
            # Update latest if needed
            if self.registry["features"][name]["latest"] == version:
                versions = self.registry["features"][name]["versions"]
                if versions:
                    self.registry["features"][name]["latest"] = versions[-1]["version"]
                else:
                    del self.registry["features"][name]
        
        self._save_registry()
        logger.info(f"Deleted feature version: {name} v{version}")
    
    def sync_to_s3(self, name: str, version: Optional[str] = None) -> None:
        """
        Sync features to S3.
        
        Args:
            name: Feature set name
            version: Version to sync (latest if None)
        """
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        try:
            import boto3
            import awswrangler as wr
        except ImportError:
            raise ImportError("AWS dependencies required for S3 sync")
        
        if version is None:
            version = self.registry["features"][name]["latest"]
        
        local_path = self.base_path / name / version
        s3_prefix = f"s3://{self.s3_bucket}/feature_store/{name}/{version}/"
        
        # Upload files
        for file_path in local_path.glob("*"):
            s3_path = f"{s3_prefix}{file_path.name}"
            
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
                wr.s3.to_parquet(df, s3_path)
            else:
                # Upload as raw file
                s3 = boto3.client("s3")
                s3.upload_file(
                    str(file_path),
                    self.s3_bucket,
                    f"feature_store/{name}/{version}/{file_path.name}"
                )
        
        logger.info(f"Synced to S3: {s3_prefix}")


__all__ = ["FeatureStore"]
