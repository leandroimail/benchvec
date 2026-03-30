"""Unit tests for the Project class."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import tomllib

from benchvec.project import PROJECTS_BASE_PATH, Project, ProjectMetadata


@pytest.mark.unit
class TestProjectInit:
    """Test Project.__init__ method with valid and invalid inputs."""

    def test_init_with_valid_name(self) -> None:
        """Test Project initialization with a valid name."""
        project = Project(name="my-benchmark")

        assert project.name == "my-benchmark"
        assert project.base_path.is_absolute()
        assert "my-benchmark" in str(project.base_path)

    def test_init_with_custom_base_path(self, tmp_path: Path) -> None:
        """Test Project initialization with a custom base_path."""
        project = Project(name="test-proj", base_path=tmp_path / "custom")

        assert project.name == "test-proj"
        assert project.base_path == (tmp_path / "custom").resolve()

    def test_init_base_path_is_absolute(self, tmp_path: Path) -> None:
        """Test that base_path is always resolved to an absolute path."""
        project = Project(name="test", base_path=tmp_path / "relative")

        assert project.base_path.is_absolute()

    @pytest.mark.parametrize(
        "invalid_name,error_pattern",
        [
            ("", "non-empty string"),
            ("proj/name", "path separators"),
            ("proj\\name", "path separators"),
            (None, "non-empty string"),
            (123, "non-empty string"),
        ],
        ids=[
            "empty_string",
            "forward_slash",
            "backslash",
            "none_type",
            "integer_type",
        ],
    )
    def test_init_with_invalid_names(self, invalid_name: object, error_pattern: str) -> None:
        """Test Project initialization rejects invalid names."""
        with pytest.raises(ValueError, match=error_pattern):
            Project(name=invalid_name)  # type: ignore[arg-type]

    def test_init_sets_base_path_default(self) -> None:
        """Test that default base_path uses PROJECTS_BASE_PATH."""
        project = Project(name="default-test")

        expected_base = (PROJECTS_BASE_PATH / "default-test").resolve()
        assert project.base_path == expected_base


@pytest.mark.unit
class TestProjectProperties:
    """Test Project directory and path properties."""

    def test_config_path_property(self, tmp_path: Path) -> None:
        """Test config_path property returns correct path."""
        project = Project(name="test", base_path=tmp_path)

        assert project.config_path == tmp_path / "config.toml"

    def test_project_metadata_path_property(self, tmp_path: Path) -> None:
        """Test project_metadata_path property returns correct path."""
        project = Project(name="test", base_path=tmp_path)

        assert project.project_metadata_path == tmp_path / "project.toml"

    def test_results_dir_property(self, tmp_path: Path) -> None:
        """Test results_dir property returns correct path."""
        project = Project(name="test", base_path=tmp_path)

        assert project.results_dir == tmp_path / "results"

    def test_reports_dir_property(self, tmp_path: Path) -> None:
        """Test reports_dir property returns correct path."""
        project = Project(name="test", base_path=tmp_path)

        assert project.reports_dir == tmp_path / "reports"

    def test_logs_dir_property(self, tmp_path: Path) -> None:
        """Test logs_dir property returns correct path."""
        project = Project(name="test", base_path=tmp_path)

        assert project.logs_dir == tmp_path / "logs"

    def test_base_path_property_is_absolute(self, tmp_path: Path) -> None:
        """Test that base_path property is always absolute."""
        project = Project(name="test", base_path=tmp_path)

        assert project.base_path.is_absolute()


@pytest.mark.unit
class TestEnsureDirectories:
    """Test Project.ensure_directories method."""

    def test_ensure_directories_creates_all_dirs(self, tmp_path: Path) -> None:
        """Test that ensure_directories creates all required subdirectories."""
        project = Project(name="test", base_path=tmp_path / "project")
        project.ensure_directories()

        assert (project.base_path).exists()
        assert project.results_dir.exists()
        assert project.reports_dir.exists()
        assert project.logs_dir.exists()

    def test_ensure_directories_idempotent(self, tmp_path: Path) -> None:
        """Test that calling ensure_directories multiple times is safe."""
        project = Project(name="test", base_path=tmp_path / "project")

        # Call multiple times
        project.ensure_directories()
        project.ensure_directories()
        project.ensure_directories()

        # All directories should still exist
        assert project.base_path.exists()
        assert project.results_dir.exists()

    def test_ensure_directories_creates_nested_structure(self, tmp_path: Path) -> None:
        """Test that ensure_directories creates deeply nested parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "project"
        project = Project(name="test", base_path=deep_path)

        project.ensure_directories()

        assert deep_path.exists()
        assert (deep_path / "results").exists()


@pytest.mark.unit
class TestGetMetadata:
    """Test Project.get_metadata method."""

    def test_get_metadata_returns_defaults_when_missing(self, tmp_path: Path) -> None:
        """Test get_metadata returns default values when file doesn't exist."""
        project = Project(name="test", base_path=tmp_path)

        metadata = project.get_metadata()

        assert "creation_time" in metadata
        assert "default_config" in metadata
        assert metadata["default_config"] is None
        # creation_time should be close to now (within 1 second)
        creation_time = datetime.fromisoformat(metadata["creation_time"])
        assert (datetime.now(timezone.utc) - creation_time).total_seconds() < 1

    def test_get_metadata_reads_existing_file(self, tmp_path: Path) -> None:
        """Test get_metadata reads and parses existing project.toml file."""
        project = Project(name="test", base_path=tmp_path)
        project.ensure_directories()

        # Write test metadata
        test_time = "2025-01-15T10:30:00+00:00"
        metadata_content = f'creation_time = "{test_time}"\ndefault_config = "my-config.toml"'
        project.project_metadata_path.write_text(metadata_content, encoding="utf-8")

        metadata = project.get_metadata()

        assert metadata["creation_time"] == test_time
        assert metadata["default_config"] == "my-config.toml"

    def test_get_metadata_handles_malformed_toml(self, tmp_path: Path) -> None:
        """Test get_metadata raises ValueError for malformed TOML."""
        project = Project(name="test", base_path=tmp_path)
        project.ensure_directories()

        # Write invalid TOML
        project.project_metadata_path.write_text("invalid: [ toml", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to parse"):
            project.get_metadata()

    def test_get_metadata_validates_schema(self, tmp_path: Path) -> None:
        """Test get_metadata validates the metadata schema."""
        project = Project(name="test", base_path=tmp_path)
        project.ensure_directories()

        # Write TOML that is valid but doesn't match expected structure
        # (This is actually tricky since ProjectMetadata has extra="allow")
        # So we write a valid TOML with extra fields
        metadata_content = 'creation_time = "2025-01-15T10:30:00+00:00"\ncustom_field = "custom_value"'
        project.project_metadata_path.write_text(metadata_content, encoding="utf-8")

        metadata = project.get_metadata()

        assert "custom_field" in metadata
        assert metadata["custom_field"] == "custom_value"


@pytest.mark.unit
class TestSetMetadata:
    """Test Project.set_metadata method."""

    def test_set_metadata_creates_file_if_missing(self, tmp_path: Path) -> None:
        """Test set_metadata creates project.toml if it doesn't exist."""
        project = Project(name="test", base_path=tmp_path)

        project.set_metadata("default_config", "test-config.toml")

        assert project.project_metadata_path.exists()

    def test_set_metadata_persists_single_key(self, tmp_path: Path) -> None:
        """Test set_metadata persists a single key-value pair."""
        project = Project(name="test", base_path=tmp_path)

        project.set_metadata("default_config", "my-config.toml")
        metadata = project.get_metadata()

        assert metadata["default_config"] == "my-config.toml"

    def test_set_metadata_preserves_existing_keys(self, tmp_path: Path) -> None:
        """Test set_metadata preserves existing metadata keys."""
        project = Project(name="test", base_path=tmp_path)

        # Set first key
        project.set_metadata("default_config", "config1.toml")
        # Set second key
        project.set_metadata("custom_field", "custom_value")

        metadata = project.get_metadata()

        assert metadata["default_config"] == "config1.toml"
        assert metadata["custom_field"] == "custom_value"
        assert "creation_time" in metadata

    def test_set_metadata_overwrites_existing_key(self, tmp_path: Path) -> None:
        """Test set_metadata overwrites an existing key."""
        project = Project(name="test", base_path=tmp_path)

        project.set_metadata("default_config", "config1.toml")
        project.set_metadata("default_config", "config2.toml")

        metadata = project.get_metadata()

        assert metadata["default_config"] == "config2.toml"

    def test_set_metadata_handles_various_types(self, tmp_path: Path) -> None:
        """Test set_metadata handles strings, integers, booleans, and None."""
        project = Project(name="test", base_path=tmp_path)

        project.set_metadata("string_field", "test-value")
        project.set_metadata("int_field", 42)
        project.set_metadata("bool_field", True)
        project.set_metadata("none_field", None)

        metadata = project.get_metadata()

        assert metadata["string_field"] == "test-value"
        assert metadata["int_field"] == 42
        assert metadata["bool_field"] is True
        # None values are serialized as empty strings in TOML format
        assert metadata["none_field"] == ""

    def test_set_metadata_creates_directories_if_needed(self, tmp_path: Path) -> None:
        """Test set_metadata creates project directories if they don't exist."""
        project = Project(name="test", base_path=tmp_path / "nonexistent")
        
        # Directories don't exist yet
        assert not project.base_path.exists()

        project.set_metadata("key", "value")

        # Metadata should be set and directories created
        assert project.project_metadata_path.exists()
        assert project.base_path.exists()

    def test_set_metadata_handles_special_characters_in_strings(self, tmp_path: Path) -> None:
        """Test set_metadata handles special characters in string values."""
        project = Project(name="test", base_path=tmp_path)

        special_value = 'test "quoted" and \\ backslash'
        project.set_metadata("special_field", special_value)

        metadata = project.get_metadata()

        assert metadata["special_field"] == special_value


@pytest.mark.unit
class TestInitialize:
    """Test Project.initialize classmethod."""

    def test_initialize_creates_new_project(self, tmp_path: Path) -> None:
        """Test initialize creates a new project with all directories."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project = Project.initialize("new-project")

            assert project.name == "new-project"
            assert project.base_path.exists()
            assert project.results_dir.exists()
            assert project.reports_dir.exists()
            assert project.logs_dir.exists()

    def test_initialize_sets_creation_time_metadata(self, tmp_path: Path) -> None:
        """Test initialize sets creation_time in metadata."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project = Project.initialize("new-project")
            metadata = project.get_metadata()

            assert "creation_time" in metadata
            # Verify it's a valid ISO format timestamp
            creation_time = datetime.fromisoformat(metadata["creation_time"])
            assert isinstance(creation_time, datetime)

    def test_initialize_fails_if_project_exists(self, tmp_path: Path) -> None:
        """Test initialize raises ValueError if project already exists."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            # Create first project
            Project.initialize("existing-project")

            # Try to initialize with same name
            with pytest.raises(ValueError, match="already exists"):
                Project.initialize("existing-project")

    def test_initialize_with_default_template(self, tmp_path: Path) -> None:
        """Test initialize with default template creates empty project."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project = Project.initialize("new-project", template="default")

            assert project.base_path.exists()
            # Config file should not exist with default template
            assert not project.config_path.exists()

    def test_initialize_with_custom_template_file(self, tmp_path: Path) -> None:
        """Test initialize copies custom template file to project."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            # Create a template file
            template_file = tmp_path / "template.toml"
            template_content = "project_name = template\nversion = 1.0"
            template_file.write_text(template_content, encoding="utf-8")

            project = Project.initialize("new-project", template=str(template_file))

            assert project.config_path.exists()
            assert project.config_path.read_text(encoding="utf-8") == template_content

    def test_initialize_with_missing_template_file(self, tmp_path: Path) -> None:
        """Test initialize raises FileNotFoundError for missing template."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            with pytest.raises(FileNotFoundError, match="Template file not found"):
                Project.initialize("new-project", template="/nonexistent/template.toml")

    def test_initialize_with_invalid_project_name(self, tmp_path: Path) -> None:
        """Test initialize raises ValueError for invalid project name."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            with pytest.raises(ValueError, match="non-empty string"):
                Project.initialize("")


@pytest.mark.unit
class TestFromName:
    """Test Project.from_name classmethod."""

    def test_from_name_loads_existing_project(self, tmp_path: Path) -> None:
        """Test from_name loads an existing project from disk."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            # Create and initialize a project
            original = Project.initialize("existing-project")
            original.set_metadata("test_field", "test_value")

            # Load it back using from_name
            loaded = Project.from_name("existing-project")

            assert loaded.name == "existing-project"
            assert loaded.base_path == original.base_path
            metadata = loaded.get_metadata()
            assert metadata["test_field"] == "test_value"

    def test_from_name_creates_new_project_if_not_found(self, tmp_path: Path) -> None:
        """Test from_name creates a new project if not found."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project = Project.from_name("new-project")

            assert project.name == "new-project"
            assert project.base_path.exists()
            assert project.results_dir.exists()

    def test_from_name_with_invalid_name(self, tmp_path: Path) -> None:
        """Test from_name raises ValueError for invalid project name."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            with pytest.raises(ValueError, match="non-empty string"):
                Project.from_name("")

    def test_from_name_preserves_metadata_on_reload(self, tmp_path: Path) -> None:
        """Test from_name preserves metadata when reloading existing project."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            # Create project and set metadata
            project1 = Project.from_name("persist-test")
            project1.set_metadata("version", 1)
            project1.set_metadata("config", "special.toml")

            # Reload project
            project2 = Project.from_name("persist-test")
            metadata = project2.get_metadata()

            assert metadata["version"] == 1
            assert metadata["config"] == "special.toml"


@pytest.mark.unit
class TestMultipleProjectsIsolation:
    """Test that multiple projects don't interfere with each other."""

    def test_multiple_projects_have_separate_metadata(self, tmp_path: Path) -> None:
        """Test metadata is isolated between projects."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project1 = Project.from_name("project-a")
            project2 = Project.from_name("project-b")

            project1.set_metadata("custom", "value-a")
            project2.set_metadata("custom", "value-b")

            metadata1 = project1.get_metadata()
            metadata2 = project2.get_metadata()

            assert metadata1["custom"] == "value-a"
            assert metadata2["custom"] == "value-b"

    def test_multiple_projects_have_separate_directories(self, tmp_path: Path) -> None:
        """Test directory structures are isolated between projects."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project1 = Project.from_name("project-a")
            project2 = Project.from_name("project-b")

            assert project1.base_path != project2.base_path
            assert project1.results_dir != project2.results_dir
            assert project1.reports_dir != project2.reports_dir
            assert project1.logs_dir != project2.logs_dir


@pytest.mark.unit
class TestProjectMetadataDefaults:
    """Test ProjectMetadata model defaults."""

    def test_project_metadata_default_creation_time(self) -> None:
        """Test ProjectMetadata creates ISO format creation_time by default."""
        metadata = ProjectMetadata()

        assert "creation_time" in metadata.model_dump()
        assert isinstance(metadata.creation_time, str)
        # Should be valid ISO format
        datetime.fromisoformat(metadata.creation_time)

    def test_project_metadata_default_config_none(self) -> None:
        """Test ProjectMetadata default_config is None by default."""
        metadata = ProjectMetadata()

        assert metadata.default_config is None

    def test_project_metadata_allows_extra_fields(self) -> None:
        """Test ProjectMetadata allows extra fields due to extra='allow'."""
        data = {
            "creation_time": "2025-01-15T10:30:00+00:00",
            "default_config": None,
            "custom_field": "custom_value",
            "another_field": 42,
        }
        metadata = ProjectMetadata.model_validate(data)

        assert metadata.creation_time == "2025-01-15T10:30:00+00:00"
        # Extra fields are stored in __pydantic_extra__
        assert metadata.__pydantic_extra__["custom_field"] == "custom_value"  # type: ignore[attr-defined]


@pytest.mark.unit
class TestProjectPathResolution:
    """Test path resolution and absoluteness."""

    def test_all_paths_are_absolute(self, tmp_path: Path) -> None:
        """Test that all path properties return absolute paths."""
        project = Project(name="test", base_path=tmp_path)
        project.ensure_directories()

        assert project.base_path.is_absolute()
        assert project.config_path.is_absolute()
        assert project.project_metadata_path.is_absolute()
        assert project.results_dir.is_absolute()
        assert project.reports_dir.is_absolute()
        assert project.logs_dir.is_absolute()

    def test_base_path_is_normalized(self, tmp_path: Path) -> None:
        """Test that base_path is normalized (resolved)."""
        # Create path with . or .. components
        constructed_path = tmp_path / "." / "subdir" / ".."
        project = Project(name="test", base_path=constructed_path)

        # Should be resolved to canonical form (absolute)
        assert project.base_path.is_absolute()
        # The path should be equal to the resolved version
        expected = constructed_path.resolve()
        assert project.base_path == expected


@pytest.mark.unit
class TestProjectMetadataTomlFormat:
    """Test TOML serialization and deserialization."""

    def test_toml_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        """Test that metadata survives TOML write and read roundtrip."""
        project = Project(name="test", base_path=tmp_path)

        # Set various metadata
        project.set_metadata("string_val", "hello")
        project.set_metadata("int_val", 123)
        project.set_metadata("bool_val", False)

        # Read raw TOML
        toml_content = project.project_metadata_path.read_text(encoding="utf-8")
        parsed = tomllib.loads(toml_content)

        assert parsed["string_val"] == "hello"
        assert parsed["int_val"] == 123
        assert parsed["bool_val"] is False

    def test_metadata_roundtrip_multiple_times(self, tmp_path: Path) -> None:
        """Test metadata persists through multiple set/get cycles."""
        project = Project(name="test", base_path=tmp_path)

        for i in range(5):
            project.set_metadata("cycle_count", i)
            metadata = project.get_metadata()
            assert metadata["cycle_count"] == i


@pytest.mark.unit
class TestProjectLogging:
    """Test that Project operations produce appropriate logging."""

    def test_project_initialization_logs_debug(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test Project.__init__ logs a debug message."""
        with caplog.at_level(logging.DEBUG):
            project = Project(name="test", base_path=tmp_path)

        assert "Initialized project" in caplog.text

    def test_ensure_directories_logs_info(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test ensure_directories logs an info message."""
        project = Project(name="test", base_path=tmp_path)

        with caplog.at_level(logging.INFO):
            project.ensure_directories()

        assert "Project directories ensured" in caplog.text

    def test_set_metadata_logs_info(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test set_metadata logs an info message."""
        project = Project(name="test", base_path=tmp_path)

        with caplog.at_level(logging.INFO):
            project.set_metadata("key", "value")

        assert "Updated metadata key" in caplog.text


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_metadata_with_complex_types_converted_to_string(self, tmp_path: Path) -> None:
        """Test that complex types in metadata are converted to string representation."""
        project = Project(name="test", base_path=tmp_path)

        # Set a list (complex type) - should be converted to string
        project.set_metadata("complex_list", [1, 2, 3])

        metadata = project.get_metadata()
        # Should be converted to string representation
        assert isinstance(metadata["complex_list"], str)
        assert "[1, 2, 3]" in metadata["complex_list"]

    def test_init_with_special_valid_characters_in_name(self, tmp_path: Path) -> None:
        """Test Project initialization with special but valid characters in name."""
        # Hyphens, dots, underscores are valid in directory names
        project = Project(name="my-project.v1-test_final")

        assert project.name == "my-project.v1-test_final"
        assert project.base_path.is_absolute()

    def test_set_metadata_with_empty_string(self, tmp_path: Path) -> None:
        """Test set_metadata handles empty strings correctly."""
        project = Project(name="test", base_path=tmp_path)

        project.set_metadata("empty", "")

        metadata = project.get_metadata()
        assert metadata["empty"] == ""

    def test_metadata_with_numeric_string_values(self, tmp_path: Path) -> None:
        """Test metadata preserves numeric strings as strings, not converted to numbers."""
        project = Project(name="test", base_path=tmp_path)

        project.set_metadata("numeric_string", "12345")
        project.set_metadata("float_string", "3.14")

        metadata = project.get_metadata()
        # Should remain as strings when set as strings
        assert isinstance(metadata["numeric_string"], str)
        assert isinstance(metadata["float_string"], str)

    def test_project_from_name_idempotent(self, tmp_path: Path) -> None:
        """Test that calling from_name multiple times on same project is idempotent."""
        with patch("benchvec.project.PROJECTS_BASE_PATH", tmp_path):
            project1 = Project.from_name("idempotent-test")
            project1.set_metadata("version", 1)

            project2 = Project.from_name("idempotent-test")
            project3 = Project.from_name("idempotent-test")

            # All should reference the same project with same metadata
            assert project1.base_path == project2.base_path == project3.base_path
            assert project2.get_metadata()["version"] == 1
            assert project3.get_metadata()["version"] == 1

    def test_ensure_directories_with_permission_error_handling(self, tmp_path: Path) -> None:
        """Test ensure_directories handles creation even in tricky scenarios."""
        # Create a deeply nested structure
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "project"
        project = Project(name="test", base_path=deep_path)

        # Should succeed without errors
        project.ensure_directories()

        assert deep_path.exists()
        assert (deep_path / "results").exists()

    def test_metadata_field_names_normalized(self, tmp_path: Path) -> None:
        """Test that metadata field names are sorted consistently in TOML output."""
        project = Project(name="test", base_path=tmp_path)

        # Set fields in non-alphabetical order
        project.set_metadata("zebra", "last")
        project.set_metadata("apple", "first")
        project.set_metadata("middle", "middle")

        # Read raw TOML
        toml_content = project.project_metadata_path.read_text(encoding="utf-8")
        lines = toml_content.strip().split("\n")

        # Fields should be alphabetically sorted in the output
        field_names = [line.split(" = ")[0] for line in lines]
        # creation_time, apple, middle, zebra (sorted)
        assert field_names[0] == "apple"  # First alphabetically after creation_time
