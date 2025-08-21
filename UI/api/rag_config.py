"""
RAG Configuration Management for Physics Assistant
Handles RAG system settings and coordination between agents
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """
    RAG system configuration settings
    """
    # Core RAG settings
    enable_rag: bool = True
    rag_api_url: str = "http://localhost:8001"
    
    # Context retrieval settings
    max_concepts: int = 5
    max_formulas: int = 3
    max_examples: int = 2
    similarity_threshold: float = 0.3
    
    # Student personalization settings
    enable_personalization: bool = True
    track_progress: bool = True
    update_student_profile: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_fallback: bool = True
    timeout_seconds: int = 30
    
    # Agent coordination settings
    enable_cross_agent_context: bool = True
    share_learning_paths: bool = True
    coordinate_concepts: bool = True
    
    # Difficulty adaptation settings
    enable_difficulty_adaptation: bool = True
    difficulty_levels: list = None
    adaptation_strategy: str = "dynamic"  # "fixed", "dynamic", "progressive"
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.difficulty_levels is None:
            self.difficulty_levels = ["beginner", "intermediate", "advanced"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        """Create configuration from dictionary"""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            assert isinstance(self.enable_rag, bool)
            assert isinstance(self.rag_api_url, str) and self.rag_api_url.strip()
            assert 1 <= self.max_concepts <= 10
            assert 1 <= self.max_formulas <= 5
            assert 0 <= self.max_examples <= 5
            assert 0.0 <= self.similarity_threshold <= 1.0
            assert self.cache_ttl_seconds > 0
            assert self.timeout_seconds > 0
            assert self.adaptation_strategy in ["fixed", "dynamic", "progressive"]
            return True
        except AssertionError as e:
            logger.error(f"RAG configuration validation failed: {e}")
            return False


class RAGConfigManager:
    """
    Manager for RAG configuration across the Physics Assistant system
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("rag_config.json")
        self._config = RAGConfig()
        self._agent_configs: Dict[str, RAGConfig] = {}
        
        # Load configuration if file exists
        self.load_config()
    
    def get_global_config(self) -> RAGConfig:
        """Get global RAG configuration"""
        return self._config
    
    def get_agent_config(self, agent_id: str) -> RAGConfig:
        """Get agent-specific RAG configuration, falling back to global"""
        if agent_id in self._agent_configs:
            return self._agent_configs[agent_id]
        return self._config
    
    def set_global_config(self, config: RAGConfig) -> bool:
        """Set global RAG configuration"""
        if config.validate():
            self._config = config
            self.save_config()
            logger.info("Global RAG configuration updated")
            return True
        return False
    
    def set_agent_config(self, agent_id: str, config: RAGConfig) -> bool:
        """Set agent-specific RAG configuration"""
        if config.validate():
            self._agent_configs[agent_id] = config
            self.save_config()
            logger.info(f"RAG configuration updated for agent: {agent_id}")
            return True
        return False
    
    def update_config(self, **kwargs) -> bool:
        """Update global configuration with specific parameters"""
        config_dict = self._config.to_dict()
        config_dict.update(kwargs)
        
        try:
            new_config = RAGConfig.from_dict(config_dict)
            return self.set_global_config(new_config)
        except Exception as e:
            logger.error(f"Failed to update RAG configuration: {e}")
            return False
    
    def update_agent_config(self, agent_id: str, **kwargs) -> bool:
        """Update agent-specific configuration with specific parameters"""
        base_config = self.get_agent_config(agent_id)
        config_dict = base_config.to_dict()
        config_dict.update(kwargs)
        
        try:
            new_config = RAGConfig.from_dict(config_dict)
            return self.set_agent_config(agent_id, new_config)
        except Exception as e:
            logger.error(f"Failed to update agent RAG configuration for {agent_id}: {e}")
            return False
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Load global config
                if "global" in data:
                    self._config = RAGConfig.from_dict(data["global"])
                
                # Load agent-specific configs
                if "agents" in data:
                    for agent_id, agent_data in data["agents"].items():
                        self._agent_configs[agent_id] = RAGConfig.from_dict(agent_data)
                
                logger.info(f"RAG configuration loaded from {self.config_file}")
                return True
            else:
                # Create default config file
                self.save_config()
                logger.info("Created default RAG configuration file")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load RAG configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            data = {
                "global": self._config.to_dict(),
                "agents": {
                    agent_id: config.to_dict() 
                    for agent_id, config in self._agent_configs.items()
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"RAG configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save RAG configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "global_config": {
                "rag_enabled": self._config.enable_rag,
                "api_url": self._config.rag_api_url,
                "personalization": self._config.enable_personalization,
                "caching": self._config.enable_caching,
                "fallback": self._config.enable_fallback
            },
            "agent_configs": list(self._agent_configs.keys()),
            "total_agents_configured": len(self._agent_configs),
            "config_file": str(self.config_file)
        }
    
    def reset_to_defaults(self) -> bool:
        """Reset all configurations to defaults"""
        self._config = RAGConfig()
        self._agent_configs.clear()
        self.save_config()
        logger.info("RAG configuration reset to defaults")
        return True
    
    def export_config(self) -> Dict[str, Any]:
        """Export all configuration for backup/sharing"""
        return {
            "global": self._config.to_dict(),
            "agents": {
                agent_id: config.to_dict() 
                for agent_id, config in self._agent_configs.items()
            },
            "metadata": {
                "exported_at": logger.info,  # Would be datetime in real implementation
                "version": "1.0.0"
            }
        }
    
    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration from backup/sharing"""
        try:
            if "global" in config_data:
                global_config = RAGConfig.from_dict(config_data["global"])
                if not global_config.validate():
                    return False
                self._config = global_config
            
            if "agents" in config_data:
                new_agent_configs = {}
                for agent_id, agent_data in config_data["agents"].items():
                    agent_config = RAGConfig.from_dict(agent_data)
                    if not agent_config.validate():
                        logger.warning(f"Invalid agent config for {agent_id}, skipping")
                        continue
                    new_agent_configs[agent_id] = agent_config
                
                self._agent_configs = new_agent_configs
            
            self.save_config()
            logger.info("RAG configuration imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import RAG configuration: {e}")
            return False


# Global config manager instance
_config_manager = None

def get_rag_config_manager() -> RAGConfigManager:
    """Get or create global RAG configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = RAGConfigManager()
    return _config_manager


def get_agent_rag_config(agent_id: str) -> RAGConfig:
    """Get RAG configuration for specific agent"""
    return get_rag_config_manager().get_agent_config(agent_id)


def update_global_rag_config(**kwargs) -> bool:
    """Update global RAG configuration"""
    return get_rag_config_manager().update_config(**kwargs)


def update_agent_rag_config(agent_id: str, **kwargs) -> bool:
    """Update agent-specific RAG configuration"""
    return get_rag_config_manager().update_agent_config(agent_id, **kwargs)