"""
CRUD operations for database models
"""
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Task, Batch, StoredFile, TaskType, TaskStatus, BatchStatus
from ..config import settings


class TaskCRUD:
    """CRUD operations for Task model"""

    @staticmethod
    async def create(db: AsyncSession, task: Task) -> Task:
        """Create a new task"""
        db.add(task)
        await db.flush()
        return task

    @staticmethod
    async def get_by_id(db: AsyncSession, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        result = await db.execute(select(Task).where(Task.id == task_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def update_status(
        db: AsyncSession,
        task_id: str,
        status: TaskStatus,
        error_message: Optional[str] = None
    ) -> Optional[Task]:
        """Update task status"""
        task = await TaskCRUD.get_by_id(db, task_id)
        if task:
            task.status = status
            if status == TaskStatus.PROCESSING and not task.started_at:
                task.started_at = datetime.utcnow()
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = datetime.utcnow()
                if task.started_at:
                    task.duration_seconds = (task.completed_at - task.started_at).total_seconds()
            if error_message:
                task.error_message = error_message
            await db.flush()
        return task

    @staticmethod
    async def update_result(
        db: AsyncSession,
        task_id: str,
        result_data: dict
    ) -> Optional[Task]:
        """Update task result data"""
        task = await TaskCRUD.get_by_id(db, task_id)
        if task:
            task.result_data = result_data
            await db.flush()
        return task

    @staticmethod
    async def get_history(
        db: AsyncSession,
        task_type: Optional[TaskType] = None,
        status: Optional[TaskStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[Task], int]:
        """Get task history with filtering and pagination"""
        query = select(Task)

        # Build filters
        filters = []
        if task_type:
            filters.append(Task.type == task_type)
        if status:
            filters.append(Task.status == status)
        if start_date:
            filters.append(Task.created_at >= start_date)
        if end_date:
            filters.append(Task.created_at <= end_date)
        if user_id:
            filters.append(Task.user_id == user_id)

        if filters:
            query = query.where(and_(*filters))

        # Get total count
        count_query = select(Task.id)
        if filters:
            count_query = count_query.where(and_(*filters))
        count_result = await db.execute(count_query)
        total = len(count_result.all())

        # Get paginated results
        query = query.order_by(Task.created_at.desc()).limit(limit).offset(offset)
        result = await db.execute(query)
        tasks = result.scalars().all()

        return list(tasks), total

    @staticmethod
    async def delete(db: AsyncSession, task_id: str) -> bool:
        """Delete a task"""
        task = await TaskCRUD.get_by_id(db, task_id)
        if task:
            await db.delete(task)
            await db.flush()
            return True
        return False


class BatchCRUD:
    """CRUD operations for Batch model"""

    @staticmethod
    async def create(db: AsyncSession, batch: Batch) -> Batch:
        """Create a new batch"""
        db.add(batch)
        await db.flush()
        return batch

    @staticmethod
    async def get_by_id(db: AsyncSession, batch_id: str) -> Optional[Batch]:
        """Get batch by ID"""
        result = await db.execute(select(Batch).where(Batch.id == batch_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def update_counts(
        db: AsyncSession,
        batch_id: str,
        completed_count: Optional[int] = None,
        failed_count: Optional[int] = None
    ) -> Optional[Batch]:
        """Update batch completion counts"""
        batch = await BatchCRUD.get_by_id(db, batch_id)
        if batch:
            if completed_count is not None:
                batch.completed_count = completed_count
            if failed_count is not None:
                batch.failed_count = failed_count

            # Update status
            if batch.completed_count + batch.failed_count >= batch.total_count:
                if batch.failed_count == batch.total_count:
                    batch.status = BatchStatus.FAILED
                elif batch.failed_count > 0:
                    batch.status = BatchStatus.PARTIAL
                else:
                    batch.status = BatchStatus.COMPLETED
                batch.completed_at = datetime.utcnow()

            await db.flush()
        return batch


class StoredFileCRUD:
    """CRUD operations for StoredFile model"""

    @staticmethod
    async def create(db: AsyncSession, file: StoredFile) -> StoredFile:
        """Create a new stored file record"""
        db.add(file)
        await db.flush()
        return file

    @staticmethod
    async def get_by_task_id(db: AsyncSession, task_id: str) -> List[StoredFile]:
        """Get all files for a task"""
        result = await db.execute(select(StoredFile).where(StoredFile.task_id == task_id))
        return list(result.scalars().all())

    @staticmethod
    async def get_expired_files(db: AsyncSession) -> List[StoredFile]:
        """Get files that have expired"""
        now = datetime.utcnow()
        result = await db.execute(
            select(StoredFile).where(
                and_(
                    StoredFile.expires_at.is_not(None),
                    StoredFile.expires_at <= now
                )
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete(db: AsyncSession, file_id: str) -> bool:
        """Delete a stored file record"""
        result = await db.execute(select(StoredFile).where(StoredFile.id == file_id))
        file = result.scalar_one_or_none()
        if file:
            await db.delete(file)
            await db.flush()
            return True
        return False
