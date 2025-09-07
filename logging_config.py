"""
Logging configuration for Trump Tweet Classifier API.

Implements hourly rotating logs with 2-week retention policy.
"""

import logging
import logging.handlers
import os
from datetime import datetime, timedelta
import glob
from pathlib import Path


class HourlyRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Custom rotating file handler that rotates every hour and cleans up old files.
    """
    
    def __init__(self, filename, retention_days=14, **kwargs):
        """
        Initialize handler with hourly rotation and retention policy.
        
        Args:
            filename: Base filename for logs
            retention_days: Number of days to keep logs (default: 14)
        """
        self.retention_days = retention_days
        super().__init__(
            filename=filename,
            when='H',  # Rotate every hour
            interval=1,
            backupCount=0,  # We'll handle cleanup manually
            **kwargs
        )
    
    def doRollover(self):
        """
        Perform log rotation and cleanup old files.
        """
        super().doRollover()
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove log files older than retention period."""
        try:
            log_dir = os.path.dirname(self.baseFilename)
            log_pattern = os.path.basename(self.baseFilename) + "*"
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for log_file in glob.glob(os.path.join(log_dir, log_pattern)):
                try:
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
                    if file_mtime < cutoff_date:
                        os.remove(log_file)
                        print(f"Removed old log file: {log_file}")
                except OSError:
                    pass  # File might be in use or already deleted
        except Exception as e:
            print(f"Error cleaning up old logs: {e}")


def setup_logging():
    """
    Configure logging for the application.
    """
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Define log format
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    formatter = logging.Formatter(log_format)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers for different log levels
    
    # Main application log (INFO and above)
    app_handler = HourlyRotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        retention_days=14
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)
    
    # Error log (ERROR and above)
    error_handler = HourlyRotatingFileHandler(
        filename=os.path.join(log_dir, "error.log"),
        retention_days=14
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Access log for HTTP requests
    access_logger = logging.getLogger("access")
    access_handler = HourlyRotatingFileHandler(
        filename=os.path.join(log_dir, "access.log"),
        retention_days=14
    )
    access_handler.setLevel(logging.INFO)
    access_formatter = logging.Formatter(
        '%(asctime)s - %(message)s'
    )
    access_handler.setFormatter(access_formatter)
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False  # Don't propagate to root logger
    
    # Database log for database operations
    db_logger = logging.getLogger("database")
    db_handler = HourlyRotatingFileHandler(
        filename=os.path.join(log_dir, "database.log"),
        retention_days=14
    )
    db_handler.setLevel(logging.INFO)
    db_handler.setFormatter(formatter)
    db_logger.addHandler(db_handler)
    db_logger.setLevel(logging.INFO)
    db_logger.propagate = False
    
    # Performance log for tracking processing times
    perf_logger = logging.getLogger("performance")
    perf_handler = HourlyRotatingFileHandler(
        filename=os.path.join(log_dir, "performance.log"),
        retention_days=14
    )
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        '%(asctime)s - %(message)s'
    )
    perf_handler.setFormatter(perf_formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False
    
    logging.info("Logging system initialized")
    logging.info(f"Log directory: {os.path.abspath(log_dir)}")
    
    return {
        'app': root_logger,
        'access': access_logger,
        'database': db_logger,
        'performance': perf_logger
    }


def get_client_ip(request) -> str:
    """
    Extract client IP address from request, handling proxies.
    """
    # Check for forwarded IP first (common with load balancers/proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP if there are multiple
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct client IP
    return getattr(request.client, 'host', 'unknown')


def log_request(request, response_time_ms: float, status_code: int, user_id: str = None):
    """
    Log HTTP request details.
    """
    access_logger = logging.getLogger("access")
    
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "Unknown")
    method = request.method
    url = str(request.url)
    
    log_message = (
        f"{client_ip} - {user_id or 'anonymous'} - "
        f"\"{method} {url}\" {status_code} - "
        f"{response_time_ms:.2f}ms - \"{user_agent}\""
    )
    
    access_logger.info(log_message)


def log_performance(operation: str, duration_ms: float, details: dict = None):
    """
    Log performance metrics.
    """
    perf_logger = logging.getLogger("performance")
    
    details_str = ""
    if details:
        details_str = " - " + " ".join([f"{k}={v}" for k, v in details.items()])
    
    perf_logger.info(f"{operation}: {duration_ms:.2f}ms{details_str}")


def log_database_operation(operation: str, table: str, duration_ms: float = None, error: str = None):
    """
    Log database operations.
    """
    db_logger = logging.getLogger("database")
    
    if error:
        db_logger.error(f"DB_ERROR - {operation} on {table}: {error}")
    else:
        duration_str = f" ({duration_ms:.2f}ms)" if duration_ms else ""
        db_logger.info(f"DB - {operation} on {table}{duration_str}")


# Initialize logging when module is imported
loggers = setup_logging()
