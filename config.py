import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Settings:
    """Application configuration sourced from environment variables."""

    # React build assets
    react_build_path: Path = Path(
        os.getenv("REACT_BUILD_PATH", Path(__file__).resolve().parent.parent / "rose-ui" / "build")
    ).resolve()

    # Auth0
    auth0_domain: Optional[str] = os.getenv("AUTH0_DOMAIN")
    auth0_audience: Optional[str] = os.getenv("AUTH0_AUDIENCE")
    auth0_client_id: Optional[str] = os.getenv("AUTH0_CLIENT_ID")

    # AWS Bedrock agent configuration
    bedrock_agent_id: Optional[str] = os.getenv("BEDROCK_AGENT_ID")
    bedrock_agent_alias_id: Optional[str] = os.getenv("BEDROCK_AGENT_ALIAS_ID")
    bedrock_region: str = os.getenv("BEDROCK_REGION", "us-east-1")

    # AWS SageMaker runtime configuration
    sagemaker_region: str = os.getenv("SAGEMAKER_REGION", os.getenv("AWS_REGION", "us-east-1"))
    sagemaker_depth_endpoint: Optional[str] = os.getenv("SAGEMAKER_DEPTH_ENDPOINT")
    sagemaker_segmentation_endpoint: Optional[str] = os.getenv("SAGEMAKER_SEGMENTATION_ENDPOINT")

    # Optional behavior flags
    require_auth_for_health: bool = os.getenv("REQUIRE_AUTH_FOR_HEALTH", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    require_auth_for_ui: bool = os.getenv("REQUIRE_AUTH_FOR_UI", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    s3_bucket_name: Optional[str] = os.getenv("S3_BUCKET_NAME")
    s3_region_name: Optional[str] = os.getenv("S3_REGION_NAME")
    s3_object_prefix: str = os.getenv("S3_OBJECT_PREFIX", "uploads/")
    s3_public_url_prefix: Optional[str] = os.getenv("S3_PUBLIC_URL_PREFIX")

    def as_flask_config(self) -> Dict[str, object]:
        data = asdict(self)
        # Normalize for Flask naming conventions
        return {
            "REACT_BUILD_PATH": str(data["react_build_path"]),
            "AUTH0_DOMAIN": data["auth0_domain"],
            "AUTH0_AUDIENCE": data["auth0_audience"],
            "AUTH0_CLIENT_ID": data["auth0_client_id"],
            "BEDROCK_AGENT_ID": data["bedrock_agent_id"],
            "BEDROCK_AGENT_ALIAS_ID": data["bedrock_agent_alias_id"],
            "BEDROCK_REGION": data["bedrock_region"],
            "SAGEMAKER_REGION": data["sagemaker_region"],
            "SAGEMAKER_DEPTH_ENDPOINT": data["sagemaker_depth_endpoint"],
            "SAGEMAKER_SEGMENTATION_ENDPOINT": data["sagemaker_segmentation_endpoint"],
            "REQUIRE_AUTH_FOR_HEALTH": data["require_auth_for_health"],
            "REQUIRE_AUTH_FOR_UI": data["require_auth_for_ui"],
            "S3_BUCKET_NAME": data["s3_bucket_name"],
            "S3_REGION_NAME": data["s3_region_name"],
            "S3_OBJECT_PREFIX": data["s3_object_prefix"],
            "S3_PUBLIC_URL_PREFIX": data["s3_public_url_prefix"],
        }

    def validate(self) -> None:
        """Validate required configuration and raise if anything critical is missing."""
        missing = []
        if not self.react_build_path.exists():
            raise FileNotFoundError(
                f"React build artifacts not found at {self.react_build_path}. "
                "Run `npm run build` in the rose-ui/ project or update REACT_BUILD_PATH."
            )

        if not self.auth0_domain:
            missing.append("AUTH0_DOMAIN")
        if not self.auth0_audience:
            missing.append("AUTH0_AUDIENCE")
        if missing:
            joined = ", ".join(missing)
            raise EnvironmentError(f"Missing required Auth0 configuration: {joined}")

        if not self.bedrock_agent_id or not self.bedrock_agent_alias_id:
            # Bedrock details are required for chat proxy
            raise EnvironmentError(
                "Missing Bedrock configuration. Set BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID."
            )

        missing_sagemaker = []
        if not self.sagemaker_depth_endpoint:
            missing_sagemaker.append("SAGEMAKER_DEPTH_ENDPOINT")
        if not self.sagemaker_segmentation_endpoint:
            missing_sagemaker.append("SAGEMAKER_SEGMENTATION_ENDPOINT")
        if missing_sagemaker:
            joined = ", ".join(missing_sagemaker)
            raise EnvironmentError(f"Missing SageMaker configuration: {joined}")

        if not self.s3_bucket_name:
            raise EnvironmentError("Missing S3 bucket configuration. Set S3_BUCKET_NAME.")

        if self.s3_object_prefix and self.s3_object_prefix.startswith("/"):
            raise EnvironmentError("S3 object prefix must not start with a leading slash.")
