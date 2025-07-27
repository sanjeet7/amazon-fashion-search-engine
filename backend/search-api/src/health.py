"""Health monitoring and system diagnostics for the search API service."""

import time
import asyncio
import psutil
from typing import Dict, Any
import logging
from pathlib import Path

from shared.models import SearchAPIConfig, HealthResponse, SearchStatsResponse, ServiceHealth
from shared.utils import setup_logger


class HealthMonitor:
    """Monitors service health and provides diagnostics."""
    
    def __init__(self, config: SearchAPIConfig, logger: logging.Logger):
        """Initialize health monitor.
        
        Args:
            config: Search API configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.start_time = time.time()
        
        # Health metrics
        self.health_metrics = {
            'service_start_time': self.start_time,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        self.logger.info("Initialized HealthMonitor")

    async def get_health_status(self, search_engine=None) -> HealthResponse:
        """Get comprehensive health status.
        
        Args:
            search_engine: Search engine instance for additional checks
            
        Returns:
            HealthResponse with system status
        """
        try:
            # Basic service health
            service_status = await self._check_service_health()
            
            # System resources
            system_status = await self._check_system_resources()
            
            # Data availability
            data_status = await self._check_data_availability()
            
            # Search engine health
            engine_status = await self._check_search_engine_health(search_engine)
            
            # Determine overall status
            overall_status = self._determine_overall_status([
                service_status, system_status, data_status, engine_status
            ])
            
            return HealthResponse(
                status=overall_status,
                timestamp=time.time(),
                uptime_seconds=time.time() - self.start_time,
                version="1.0.0",
                checks={
                    'service': service_status.value,
                    'system': system_status.value,
                    'data': data_status.value,
                    'search_engine': engine_status.value
                },
                details={
                    'system_resources': await self._get_system_details(),
                    'data_files': await self._get_data_file_status(),
                    'performance_metrics': self._get_performance_metrics()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status=ServiceHealth.UNHEALTHY,
                timestamp=time.time(),
                uptime_seconds=time.time() - self.start_time,
                version="1.0.0",
                checks={'error': 'health_check_failed'},
                details={'error': str(e)}
            )

    async def _check_service_health(self) -> ServiceHealth:
        """Check basic service health.
        
        Returns:
            Service health status
        """
        try:
            # Check if service has been running for minimum time
            if time.time() - self.start_time < 10:  # 10 seconds
                return ServiceHealth.STARTING
            
            # Check basic functionality
            await asyncio.sleep(0.001)  # Basic async test
            
            return ServiceHealth.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Service health check failed: {e}")
            return ServiceHealth.UNHEALTHY

    async def _check_system_resources(self) -> ServiceHealth:
        """Check system resource availability.
        
        Returns:
            System health status
        """
        try:
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return ServiceHealth.DEGRADED
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                return ServiceHealth.DEGRADED
            
            # Disk check
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                return ServiceHealth.DEGRADED
            
            return ServiceHealth.HEALTHY
            
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return ServiceHealth.DEGRADED

    async def _check_data_availability(self) -> ServiceHealth:
        """Check data file availability.
        
        Returns:
            Data health status
        """
        try:
            required_files = [
                self.config.processed_data_dir / "processed_sample.parquet",
                self.config.embeddings_cache_dir / "embeddings.npz",
                self.config.embeddings_cache_dir / "faiss_index.bin"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                self.logger.warning(f"Missing data files: {missing_files}")
                return ServiceHealth.UNHEALTHY
            
            return ServiceHealth.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Data availability check failed: {e}")
            return ServiceHealth.UNHEALTHY

    async def _check_search_engine_health(self, search_engine=None) -> ServiceHealth:
        """Check search engine health.
        
        Args:
            search_engine: Search engine instance
            
        Returns:
            Search engine health status
        """
        try:
            if search_engine is None:
                return ServiceHealth.DEGRADED
            
            # Check if search engine is initialized
            if search_engine.index is None or search_engine.products_df is None:
                return ServiceHealth.UNHEALTHY
            
            # Check data integrity
            if len(search_engine.products_df) == 0:
                return ServiceHealth.UNHEALTHY
            
            return ServiceHealth.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Search engine health check failed: {e}")
            return ServiceHealth.UNHEALTHY

    def _determine_overall_status(self, statuses: list) -> ServiceHealth:
        """Determine overall health from component statuses.
        
        Args:
            statuses: List of component health statuses
            
        Returns:
            Overall health status
        """
        if ServiceHealth.UNHEALTHY in statuses:
            return ServiceHealth.UNHEALTHY
        elif ServiceHealth.DEGRADED in statuses:
            return ServiceHealth.DEGRADED
        elif ServiceHealth.STARTING in statuses:
            return ServiceHealth.STARTING
        else:
            return ServiceHealth.HEALTHY

    async def _get_system_details(self) -> Dict[str, Any]:
        """Get detailed system resource information.
        
        Returns:
            System resource details
        """
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent
                },
                'cpu': {
                    'cores': psutil.cpu_count(),
                    'usage_percent': psutil.cpu_percent(interval=0.1)
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_percent': disk.percent
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get system details: {e}")
            return {'error': str(e)}

    async def _get_data_file_status(self) -> Dict[str, Any]:
        """Get data file status information.
        
        Returns:
            Data file status details
        """
        try:
            file_status = {}
            
            # Check processed data
            processed_file = self.config.processed_data_dir / "processed_sample.parquet"
            if processed_file.exists():
                stat = processed_file.stat()
                file_status['processed_data'] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024**2),
                    'modified_time': stat.st_mtime
                }
            else:
                file_status['processed_data'] = {'exists': False}
            
            # Check embeddings
            embeddings_file = self.config.embeddings_cache_dir / "embeddings.npz"
            if embeddings_file.exists():
                stat = embeddings_file.stat()
                file_status['embeddings'] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024**2),
                    'modified_time': stat.st_mtime
                }
            else:
                file_status['embeddings'] = {'exists': False}
            
            # Check FAISS index
            index_file = self.config.embeddings_cache_dir / "faiss_index.bin"
            if index_file.exists():
                stat = index_file.stat()
                file_status['faiss_index'] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024**2),
                    'modified_time': stat.st_mtime
                }
            else:
                file_status['faiss_index'] = {'exists': False}
            
            return file_status
            
        except Exception as e:
            self.logger.error(f"Failed to get data file status: {e}")
            return {'error': str(e)}

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Performance metrics
        """
        return {
            'total_requests': self.health_metrics['total_requests'],
            'successful_requests': self.health_metrics['successful_requests'],
            'failed_requests': self.health_metrics['failed_requests'],
            'success_rate': (
                self.health_metrics['successful_requests'] / max(1, self.health_metrics['total_requests'])
            ),
            'avg_response_time_ms': self.health_metrics['avg_response_time']
        }

    async def get_search_statistics(self, search_engine=None) -> SearchStatsResponse:
        """Get comprehensive search statistics.
        
        Args:
            search_engine: Search engine instance
            
        Returns:
            Search statistics response
        """
        try:
            # Basic stats
            basic_stats = self._get_performance_metrics()
            
            # Search engine stats
            engine_stats = {}
            if search_engine:
                engine_stats = {
                    'total_products': len(search_engine.products_df) if search_engine.products_df is not None else 0,
                    'index_size': search_engine.index.ntotal if search_engine.index else 0,
                    'embedding_dimension': search_engine.embeddings.shape[1] if search_engine.embeddings is not None else 0,
                    'search_performance': search_engine.search_stats
                }
            
            # System stats
            system_stats = await self._get_system_details()
            
            return SearchStatsResponse(
                service_stats=basic_stats,
                search_engine_stats=engine_stats,
                system_stats=system_stats,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get search statistics: {e}")
            raise

    def record_request(self, success: bool, response_time: float) -> None:
        """Record request metrics.
        
        Args:
            success: Whether request was successful
            response_time: Response time in seconds
        """
        self.health_metrics['total_requests'] += 1
        
        if success:
            self.health_metrics['successful_requests'] += 1
        else:
            self.health_metrics['failed_requests'] += 1
        
        # Update rolling average response time
        current_avg = self.health_metrics['avg_response_time']
        total_requests = self.health_metrics['total_requests']
        
        new_avg = ((current_avg * (total_requests - 1)) + (response_time * 1000)) / total_requests
        self.health_metrics['avg_response_time'] = new_avg 