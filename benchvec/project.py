"""Project management for BenchVec benchmark projects."""

from __future__ import annotations

import logging
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Base directory for all projects
PROJECTS_BASE_PATH = Path(".benchmarks") / "projects"


def _simple_toml_dump(data: dict[str, Any]) -> str:
    """Convert a simple metadata dictionary to TOML format.

    This is a simple TOML writer for basic key-value pairs in the metadata.
    Handles strings, booleans, integers, and None values.

    Args:
        data: Dictionary to convert to TOML format.

    Returns:
        TOML-formatted string suitable for writing to a file.
    """
    lines: list[str] = []
    for key, value in sorted(data.items()):
        if isinstance(value, str):
            # Escape quotes and newlines in strings
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        elif value is None:
            lines.append(f'{key} = ""')
        else:
            # For complex types, convert to string representation
            lines.append(f'{key} = "{str(value)}"')
    return "\n".join(lines) + "\n"


class ProjectMetadata(BaseModel):
    """Metadata model for project.toml configuration."""

    model_config = ConfigDict(extra="allow")

    creation_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    default_config: str | None = None


class Project:
    """Core project class for managing benchmark projects.

    A project organizes benchmark configurations, results, and reports in a structured
    directory hierarchy. Each project has a name, metadata file, and dedicated directories
    for results, reports, and logs.
    """

    def __init__(
        self,
        name: str,
        base_path: Path | None = None,
    ) -> None:
        """Initialize a Project instance.

        Args:
            name: Project name (e.g., "my-benchmark"). Must be a valid directory name.
            base_path: Optional resolved absolute path to project folder. If None,
                uses the default PROJECTS_BASE_PATH / name.

        Raises:
            ValueError: If name is empty or contains invalid characters.
        """
        if not name or not isinstance(name, str):
            msg = "Project name must be a non-empty string"
            raise ValueError(msg)

        if any(c in name for c in ("\\", "/")):
            msg = 'Project name cannot contain path separators ("/" or "\\")'
            raise ValueError(msg)

        self.name: str = name
        self.base_path: Path = (base_path or PROJECTS_BASE_PATH / name).resolve()

        logger.debug(f"Initialized project '{name}' at {self.base_path}")

    @property
    def config_path(self) -> Path:
        """Path to optional config.toml template file in project root."""
        return self.base_path / "config.toml"

    @property
    def project_metadata_path(self) -> Path:
        """Path to project.toml metadata file in project root."""
        return self.base_path / "project.toml"

    @property
    def results_dir(self) -> Path:
        """Path to results directory."""
        return self.base_path / "results"

    @property
    def reports_dir(self) -> Path:
        """Path to reports directory."""
        return self.base_path / "reports"

    @property
    def logs_dir(self) -> Path:
        """Path to logs directory."""
        return self.base_path / "logs"

    @property
    def charts_dir(self) -> Path:
        """Path to charts directory within reports."""
        return self.reports_dir / "charts"

    def find_config(self, config_name: str | None = None) -> Path:
        """Locate a benchmark configuration file for this project.

        Lookup order:
        1. ``config_name`` resolved inside the project directory
        2. ``config_name`` resolved inside the repo-wide ``configs/`` directory
        3. the project's own ``config.toml`` (written by ``init --template``)
        4. any TOML/YAML file in the project directory
        5. any TOML/YAML file in the repo-wide ``configs/`` directory

        Raises:
            FileNotFoundError: If no candidate file exists.
        """

        candidates: list[Path] = []
        if config_name:
            candidates.append(self.base_path / config_name)
            candidates.append(Path("configs") / config_name)
        candidates.append(self.config_path)
        # project.toml holds project metadata, not a benchmark config.
        candidates.extend(
            path
            for path in sorted(self.base_path.glob("*.toml"))
            if path != self.project_metadata_path
        )
        candidates.extend(sorted(self.base_path.glob("*.yaml")))
        candidates.extend(sorted(Path("configs").glob("*.toml")))
        candidates.extend(sorted(Path("configs").glob("*.yaml")))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No configuration file found for project '{self.name}' "
            f"(looked in {self.base_path} and configs/)"
        )

    def ensure_directories(self) -> None:
        """Create all required project directories if they don't exist.

        Creates the project base path and all subdirectories: results, reports, charts, and logs.
        """
        directories = [
            self.base_path,
            self.results_dir,
            self.reports_dir,
            self.charts_dir,
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

        logger.info(f"Project directories ensured for '{self.name}'")

    def get_metadata(self) -> dict[str, Any]:
        """Read and parse project.toml metadata file.

        Returns a dictionary with project metadata. If the metadata file doesn't exist,
        returns a dictionary with default values (creation_time set to current time).

        Returns:
            Dictionary containing project metadata with at least:
            - creation_time: ISO format timestamp
            - default_config: Optional path to default config

        Raises:
            ValueError: If project.toml file is malformed or not a valid mapping.
        """
        if not self.project_metadata_path.exists():
            logger.debug(
                f"Metadata file not found for project '{self.name}', "
                f"using defaults"
            )
            return ProjectMetadata().model_dump()

        try:
            content = self.project_metadata_path.read_text(encoding="utf-8")
            raw_metadata = tomllib.loads(content)

            if not isinstance(raw_metadata, dict):
                msg = f"project.toml must contain a mapping, not {type(raw_metadata)}"
                raise ValueError(msg)

            # Validate and merge with defaults
            metadata = ProjectMetadata.model_validate(raw_metadata)
            logger.debug(f"Loaded metadata for project '{self.name}'")
            return metadata.model_dump()

        except tomllib.TOMLDecodeError as e:
            msg = f"Failed to parse project.toml for '{self.name}': {e}"
            raise ValueError(msg) from e

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Update a single metadata key in project.toml.

        Reads the current metadata, updates the specified key, and writes back to disk.
        If a key doesn't exist in the default metadata schema, it will still be written
        (due to extra="allow" in ProjectMetadata).

        Args:
            key: Metadata key to set (e.g., "default_config", "custom_field").
            value: Value to set for the key. Must be JSON-serializable.

        Raises:
            ValueError: If the metadata file is malformed or cannot be written.
        """
        try:
            # Read current metadata or start with defaults
            current = self.get_metadata()

            # Update with new value
            current[key] = value

            # Validate updated metadata
            ProjectMetadata.model_validate(current)

            # Ensure parent directory exists
            self.ensure_directories()

            # Write metadata back to disk using TOML format
            metadata_toml = _simple_toml_dump(current)
            self.project_metadata_path.write_text(metadata_toml, encoding="utf-8")

            logger.info(f"Updated metadata key '{key}' for project '{self.name}'")

        except ValueError as e:
            msg = f"Failed to set metadata for project '{self.name}': {e}"
            raise ValueError(msg) from e

    @classmethod
    def initialize(
        cls,
        name: str,
        template: str = "default",
    ) -> Project:
        """Create a new project with the given name and optional template.

        Creates the project directory structure, initializes metadata file with
        creation_time, and optionally copies a template configuration.

        Args:
            name: Project name (e.g., "my-benchmark").
            template: Template name for initial configuration ("default" or a path).
                "default" creates an empty project; a path to a TOML/YAML file
                is copied into the project as its config.toml.

        Returns:
            Initialized Project instance with directories created.

        Raises:
            ValueError: If project name is invalid or project already exists.
            FileNotFoundError: If template file doesn't exist.
        """
        project = cls(name)

        if project.base_path.exists():
            logger.warning(f"Project '{name}' already exists at {project.base_path}")
            msg = f"Project '{name}' already exists"
            raise ValueError(msg)

        # Create directories
        project.ensure_directories()

        # Initialize metadata with creation_time
        metadata = ProjectMetadata()
        project.set_metadata("creation_time", metadata.creation_time)

        # Handle template configuration
        if template != "default":
            template_path = Path(template)
            if not template_path.exists():
                msg = f"Template file not found: {template}"
                raise FileNotFoundError(msg)
            # Copy template to project config
            project.config_path.write_text(
                template_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            logger.info(f"Applied template '{template}' to project '{name}'")

        logger.info(f"Created project '{name}' at {project.base_path}")
        return project

    @classmethod
    def from_name(cls, name: str) -> Project:
        """Load existing project or create with defaults if not found.

        Looks for a project with the given name in PROJECTS_BASE_PATH. If found,
        loads it. If not found, creates a new project with default settings.

        Args:
            name: Project name to load or create.

        Returns:
            Project instance either loaded from disk or newly initialized.

        Raises:
            ValueError: If project name is invalid.
        """
        project_path = PROJECTS_BASE_PATH / name

        if project_path.exists():
            logger.info(f"Loading existing project '{name}' from {project_path}")
            project = cls(name, base_path=project_path)
            return project

        logger.info(f"Project '{name}' not found, creating with defaults")
        return cls.initialize(name)
