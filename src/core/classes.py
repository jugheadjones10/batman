"""Class management utilities for renaming and merging classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.project import Project


@dataclass
class RenameResult:
    """Result of a class rename operation."""

    old_name: str
    new_name: str
    message: str


@dataclass
class MergeResult:
    """Result of a class merge operation."""

    source_classes: list[str]
    target_class: str
    annotations_updated: int
    classes_removed: list[str]
    message: str


class ClassManager:
    """
    Manage class labels within a project.

    Provides operations for renaming and merging classes, including
    updating all associated annotations.
    """

    def __init__(self, project: "Project"):
        self.project = project

    def rename_class(self, old_name: str, new_name: str) -> RenameResult:
        """
        Rename a class.

        This only changes the class name in the project configuration.
        Annotations are not modified since they reference classes by index.

        Args:
            old_name: Current class name
            new_name: New class name

        Returns:
            RenameResult with operation details

        Raises:
            ValueError: If old_name doesn't exist or new_name already exists
        """
        if old_name not in self.project.classes:
            raise ValueError(f"Class '{old_name}' not found")

        if new_name in self.project.classes and new_name != old_name:
            raise ValueError(f"Class '{new_name}' already exists")

        if old_name == new_name:
            return RenameResult(
                old_name=old_name,
                new_name=new_name,
                message="No change needed",
            )

        # Update class name in list
        idx = self.project.classes.index(old_name)
        self.project.classes[idx] = new_name

        # Update source tracking
        if old_name in self.project.class_sources:
            self.project.class_sources[new_name] = self.project.class_sources.pop(old_name)

        # Save project
        self.project.save()

        return RenameResult(
            old_name=old_name,
            new_name=new_name,
            message=f"Renamed '{old_name}' to '{new_name}'",
        )

    def merge_classes(
        self,
        source_classes: list[str],
        target_class: str,
    ) -> MergeResult:
        """
        Merge multiple classes into one.

        All annotations with source class IDs are updated to use the target class ID.
        Source classes are removed from the project, and all class IDs are remapped.

        Args:
            source_classes: List of class names to merge FROM
            target_class: Class name to merge INTO

        Returns:
            MergeResult with operation details

        Raises:
            ValueError: If any class doesn't exist

        Example:
            # Merge "container-spreader" and "spreader-old" into "spreader"
            manager.merge_classes(
                source_classes=["container-spreader", "spreader-old", "spreader"],
                target_class="spreader"
            )
        """
        # Validate all classes exist
        for cls in source_classes:
            if cls not in self.project.classes:
                raise ValueError(f"Class '{cls}' not found")

        if target_class not in self.project.classes:
            raise ValueError(f"Target class '{target_class}' not found")

        # Get class IDs (before any modifications)
        target_id = self.project.classes.index(target_class)
        source_ids = [
            self.project.classes.index(cls)
            for cls in source_classes
            if cls != target_class
        ]

        if not source_ids:
            return MergeResult(
                source_classes=source_classes,
                target_class=target_class,
                annotations_updated=0,
                classes_removed=[],
                message="No classes to merge",
            )

        # Load annotations
        annotations = self.project.load_annotations()
        updated_count = 0

        # Step 1: Update all source class annotations to target class
        for ann_id, ann in annotations.items():
            if ann.get("class_label_id") in source_ids:
                annotations[ann_id]["class_label_id"] = target_id
                updated_count += 1

        # Step 2: Build new class list (remove source classes, keep target)
        new_classes = [
            cls for cls in self.project.classes
            if cls not in source_classes or cls == target_class
        ]

        # Step 3: Build old_id -> new_id mapping for remapping
        old_to_new = {}
        for old_id, old_name in enumerate(self.project.classes):
            if old_name in new_classes:
                new_id = new_classes.index(old_name)
                old_to_new[old_id] = new_id

        # Step 4: Remap all annotation class IDs
        for ann_id, ann in annotations.items():
            old_class_id = ann.get("class_label_id", 0)
            if old_class_id in old_to_new:
                annotations[ann_id]["class_label_id"] = old_to_new[old_class_id]

        # Step 5: Save annotations
        self.project.save_annotations(annotations)

        # Step 6: Update project classes
        classes_removed = [cls for cls in source_classes if cls != target_class]
        self.project.classes = new_classes

        # Step 7: Update class sources (remove merged classes)
        for cls in classes_removed:
            if cls in self.project.class_sources:
                del self.project.class_sources[cls]

        # Step 8: Save project
        self.project.save()

        return MergeResult(
            source_classes=source_classes,
            target_class=target_class,
            annotations_updated=updated_count,
            classes_removed=classes_removed,
            message=f"Merged {len(classes_removed)} classes into '{target_class}'",
        )

    def list_classes(self) -> list[dict]:
        """
        List all classes with their metadata.

        Returns:
            List of dicts with class info (name, index, source, annotation_count)
        """
        annotations = self.project.load_annotations()

        # Count annotations per class
        class_counts: dict[int, int] = {}
        for ann in annotations.values():
            class_id = ann.get("class_label_id", 0)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        result = []
        for idx, name in enumerate(self.project.classes):
            result.append({
                "name": name,
                "index": idx,
                "source": self.project.class_sources.get(name, "unknown"),
                "annotation_count": class_counts.get(idx, 0),
            })

        return result
