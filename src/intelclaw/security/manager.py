"""
Security Manager - Authentication and permissions.

Handles user authentication, tool permissions, and audit logging.
"""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from intelclaw.config.manager import ConfigManager


class SecurityManager:
    """
    Security manager for IntelCLaw.
    
    Features:
    - Permission management
    - Audit logging
    - Sensitive operation confirmation
    - Credential storage (Windows Credential Manager)
    """
    
    # Default permissions
    DEFAULT_PERMISSIONS = {
        "read": True,
        "write": True,
        "execute": True,
        "network": True,
        "sensitive": False,  # Requires explicit grant
    }
    
    def __init__(self, config: "ConfigManager"):
        """
        Initialize security manager.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self._permissions: Dict[str, bool] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._audit_file: Optional[Path] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize security manager."""
        logger.info("Initializing security manager...")
        
        # Load permissions
        self._permissions = self.DEFAULT_PERMISSIONS.copy()
        security_config = self.config.get("security", {})
        
        # Set up audit logging
        audit_path = security_config.get("audit_log_path", "data/audit.log")
        self._audit_file = Path(audit_path)
        self._audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        logger.info("Security manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown security manager."""
        # Flush audit log
        await self._flush_audit_log()
        logger.info("Security manager shutdown complete")
    
    async def has_permission(self, permission: str) -> bool:
        """
        Check if a permission is granted.
        
        Args:
            permission: Permission name
            
        Returns:
            True if permitted
        """
        return self._permissions.get(permission, False)
    
    async def grant_permission(self, permission: str) -> None:
        """Grant a permission."""
        self._permissions[permission] = True
        await self._log_audit("permission_granted", {"permission": permission})
        logger.info(f"Permission granted: {permission}")
    
    async def revoke_permission(self, permission: str) -> None:
        """Revoke a permission."""
        self._permissions[permission] = False
        await self._log_audit("permission_revoked", {"permission": permission})
        logger.info(f"Permission revoked: {permission}")
    
    async def request_confirmation(
        self,
        action: str,
        details: str,
        timeout: int = 30
    ) -> bool:
        """
        Request user confirmation for sensitive operation.
        
        Args:
            action: Action description
            details: Detailed explanation
            timeout: Timeout in seconds
            
        Returns:
            True if confirmed
        """
        # In a real implementation, this would show a UI dialog
        # For now, auto-approve non-destructive actions
        
        destructive_keywords = ["delete", "remove", "format", "shutdown"]
        is_destructive = any(kw in action.lower() for kw in destructive_keywords)
        
        if is_destructive:
            logger.warning(f"Destructive action requested: {action}")
            await self._log_audit("confirmation_required", {
                "action": action,
                "details": details,
                "auto_approved": False,
            })
            return False  # Require explicit confirmation
        
        await self._log_audit("confirmation_auto_approved", {
            "action": action,
            "details": details,
        })
        return True
    
    async def store_credential(
        self,
        name: str,
        credential: str
    ) -> bool:
        """
        Store a credential securely.
        
        Args:
            name: Credential name
            credential: Credential value
            
        Returns:
            True if successful
        """
        try:
            import keyring
            keyring.set_password("IntelCLaw", name, credential)
            await self._log_audit("credential_stored", {"name": name})
            return True
        except Exception as e:
            logger.error(f"Failed to store credential: {e}")
            return False
    
    async def get_credential(self, name: str) -> Optional[str]:
        """
        Retrieve a stored credential.
        
        Args:
            name: Credential name
            
        Returns:
            Credential value or None
        """
        try:
            import keyring
            return keyring.get_password("IntelCLaw", name)
        except Exception as e:
            logger.warning(f"Failed to retrieve credential: {e}")
            return None
    
    async def delete_credential(self, name: str) -> bool:
        """Delete a stored credential."""
        try:
            import keyring
            keyring.delete_password("IntelCLaw", name)
            await self._log_audit("credential_deleted", {"name": name})
            return True
        except Exception as e:
            logger.warning(f"Failed to delete credential: {e}")
            return False
    
    async def _log_audit(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }
        
        self._audit_log.append(event)
        
        # Write to file periodically
        if len(self._audit_log) >= 10:
            await self._flush_audit_log()
    
    async def _flush_audit_log(self) -> None:
        """Flush audit log to file."""
        if not self._audit_file or not self._audit_log:
            return
        
        try:
            with self._audit_file.open("a", encoding="utf-8") as f:
                for event in self._audit_log:
                    f.write(json.dumps(event) + "\n")
            
            self._audit_log.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit log: {e}")
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events."""
        return self._audit_log[-limit:]
    
    def validate_path(self, path: str) -> bool:
        """
        Validate that a path is allowed.
        
        Args:
            path: Path to validate
            
        Returns:
            True if allowed
        """
        path_obj = Path(path).resolve()
        
        # Check against allowed directories
        allowed = self.config.get("security.allowed_directories", [])
        
        if not allowed:
            return True  # No restrictions
        
        for allowed_dir in allowed:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_obj.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False
