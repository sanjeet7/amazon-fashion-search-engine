import React from 'react';
import { CheckCircle, AlertCircle, XCircle, Clock } from 'lucide-react';
import './StatusIndicator.css';

const StatusIndicator = ({ health }) => {
  if (!health) {
    return (
      <div className="status-indicator status-loading">
        <Clock className="status-icon loading-pulse" size={16} />
        <span>Checking API status...</span>
      </div>
    );
  }

  const getStatusDisplay = () => {
    switch (health.status) {
      case 'healthy':
        return {
          icon: <CheckCircle size={16} />,
          text: `API Ready • ${health.total_products || 0} products indexed`,
          className: 'status-healthy'
        };
      case 'initializing':
        return {
          icon: <Clock size={16} />,
          text: 'API Initializing • Please wait...',
          className: 'status-initializing'
        };
      case 'error':
        return {
          icon: <XCircle size={16} />,
          text: 'API Unavailable • Check backend service',
          className: 'status-error'
        };
      default:
        return {
          icon: <AlertCircle size={16} />,
          text: 'Unknown Status',
          className: 'status-warning'
        };
    }
  };

  const { icon, text, className } = getStatusDisplay();

  return (
    <div className={`status-indicator ${className}`}>
      {icon}
      <span>{text}</span>
      {health.status === 'healthy' && health.avg_search_time_ms && (
        <span className="status-detail">
          Avg: {Math.round(health.avg_search_time_ms)}ms
        </span>
      )}
    </div>
  );
};

export default StatusIndicator;