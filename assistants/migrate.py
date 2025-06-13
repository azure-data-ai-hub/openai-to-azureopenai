import os
import sys
import time
import tempfile
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    logger.error("Required packages not installed. Install with: pip install openai")
    sys.exit(1)

class AssistantMigrator:
    def __init__(
        self,
        openai_api_key: str,
        azure_api_key: str,
        azure_endpoint: str,
        azure_api_version: str = "2023-12-01-preview"
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.azure_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint
        )
        
        # Validation
        try:
            self.openai_client.models.list()
            logger.info("Successfully connected to OpenAI API")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {e}")
            sys.exit(1)
            
        try:
            self.azure_client.models.list()
            logger.info("Successfully connected to Azure OpenAI API")
        except Exception as e:
            logger.error(f"Failed to connect to Azure OpenAI API: {e}")
            sys.exit(1)
    
    def _download_file(self, file_id: str) -> str:
        """Download a file from OpenAI and return a temporary path."""
        logger.info(f"Downloading file {file_id} from OpenAI …")

        try:
            # Recommended pattern in openai-python ≥1.14
            with self.openai_client.files.content(file_id) as remote_fp:
                file_bytes = remote_fp.read()
        except Exception as exc:
            logger.error(f"Unable to download file {file_id}: {exc}")
            raise

        meta = self.openai_client.files.retrieve(file_id)
        filename = meta.filename or f"{file_id}.bin"
        file_path = os.path.join(tempfile.gettempdir(), filename)

        with open(file_path, "wb") as local_fp:
            local_fp.write(file_bytes)

        logger.info(f"Saved to {file_path}")
        return file_path
    
    def _upload_file_to_azure(self, file_path: str) -> str:
        """Upload a file to Azure OpenAI and return the file ID"""
        logger.info(f"Uploading file {file_path} to Azure OpenAI...")
        
        try:
            with open(file_path, "rb") as f:
                response = self.azure_client.files.create(
                    file=f,
                    purpose="assistants"
                )
            logger.info(f"File uploaded to Azure OpenAI with ID: {response.id}")
            return response.id
        except Exception as e:
            logger.error(f"Failed to upload file to Azure OpenAI: {e}")
            return None
    
    def get_openai_assistants(self) -> List[Any]:
        """Stream all assistants from OpenAI (no 100-item cap)."""
        logger.info("Fetching assistants from OpenAI …")

        try:
            # auto_paging_iter() yields every item across pages
            return list(
                self.openai_client.beta.assistants.list(limit=100).auto_paging_iter()
            )
        except Exception as exc:
            logger.error(f"Assistant fetch failed: {exc}")
            return []
    
    def get_assistant_details(self, assistant_id: str) -> Dict[str, Any]:
        """Get detailed information about an assistant"""
        logger.info(f"Fetching details for assistant {assistant_id}...")
        try:
            assistant = self.openai_client.beta.assistants.retrieve(assistant_id)
            
            # Get file details
            file_details = []
            for file_id in assistant.file_ids:
                file_path = self._download_file(file_id)
                file_details.append({
                    "id": file_id,
                    "path": file_path
                })
            
            return {
                "id": assistant.id,
                "name": assistant.name,
                "instructions": assistant.instructions,
                "model": assistant.model,
                "tools": assistant.tools,
                "file_details": file_details
            }
        except Exception as e:
            logger.error(f"Failed to fetch details for assistant {assistant_id}: {e}")
            return {}
    
    def create_azure_assistant(self, details: Dict[str, Any]) -> Optional[str]:
        """Create an assistant on Azure OpenAI based on the provided details"""
        logger.info(f"Creating assistant '{details.get('name')}' on Azure OpenAI...")
        
        # Handle files
        azure_file_ids = []
        for file_detail in details.get("file_details", []):
            azure_file_id = self._upload_file_to_azure(file_detail["path"])
            if azure_file_id:
                azure_file_ids.append(azure_file_id)
        
        # Find an appropriate model / deployment available in Azure
        try:
            deployments = self.azure_client.models.list().data
            deployment_ids = [d.id for d in deployments]

            target_model = details.get("model")
            if target_model not in deployment_ids:
                # fall-back logic
                gpt4 = next((d for d in deployment_ids if "gpt-4" in d), None)
                target_model = gpt4 or (deployment_ids[0] if deployment_ids else None)

            if not target_model:
                logger.error("No deployments available in Azure OpenAI.")
                return None
            if target_model != details.get("model"):
                logger.info(f"Using Azure deployment '{target_model}' instead of '{details.get('model')}'")

            response = self.azure_client.beta.assistants.create(
                name=details.get("name"),
                instructions=details.get("instructions"),
                model=target_model,               # deployment id
                tools=details.get("tools", []),
                file_ids=azure_file_ids,
            )
            logger.info(f"Created Azure assistant {response.id}")
            return response.id
        except Exception as exc:
            logger.error(f"Azure assistant creation failed: {exc}")
            return None
    
    def migrate_all_assistants(self) -> Dict[str, str]:
        """Migrate all assistants from OpenAI to Azure OpenAI."""
        migration_map: Dict[str, str] = {}

        # iterate directly over the paginated cursor to reduce RAM usage
        cursor = self.openai_client.beta.assistants.list(limit=100).auto_paging_iter()
        for assistant in cursor:
            logger.info(f"Migrating assistant: {getattr(assistant, 'name', assistant.id)}")

            details = self.get_assistant_details(assistant.id)
            if not details:
                continue

            azure_assistant_id = self.create_azure_assistant(details)
            if azure_assistant_id:
                migration_map[assistant.id] = azure_assistant_id
                logger.info(f"Successfully migrated {assistant.id} → {azure_assistant_id}")
            else:
                logger.error(f"Failed to migrate assistant {assistant.id}")

            time.sleep(1)  # minimal throttle; tune if you hit rate limits

        return migration_map

def main():
    # Check for environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not all([openai_api_key, azure_api_key, azure_endpoint]):
        logger.error(
            "Please set the following environment variables:\n"
            "- OPENAI_API_KEY\n"
            "- AZURE_OPENAI_API_KEY\n"
            "- AZURE_OPENAI_ENDPOINT"
        )
        sys.exit(1)
    
    migrator = AssistantMigrator(
        openai_api_key=openai_api_key,
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint
    )
    
    migration_results = migrator.migrate_all_assistants()
    
    logger.info("Migration completed!")
    logger.info("Migration summary:")
    for source_id, target_id in migration_results.items():
        logger.info(f"- {source_id} → {target_id}")
    
    logger.info(f"Successfully migrated {len(migration_results)} assistants")

if __name__ == "__main__":
    main()