/**
 * ErrorBoundary Component
 * 
 * Catches React errors in child component tree and logs them
 * to our debugging system.
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import debugService from '../utils/debug-service';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { 
      hasError: false,
      error: null 
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    // Update state to show fallback UI
    return { 
      hasError: true,
      error 
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error to our debug service
    debugService.logClientError(error, {
      componentStack: errorInfo.componentStack,
      location: window.location.href
    }).catch(e => {
      console.error('Failed to log error to debug service:', e);
    });
    
    // Also log to console for development
    console.error('React Error:', error);
    console.error('Component Stack:', errorInfo.componentStack);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      // Return fallback UI if provided, otherwise show default error message
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      return (
        <div className="p-4 bg-red-50 border border-red-300 rounded-md">
          <h2 className="text-lg font-semibold text-red-700 mb-2">
            Something went wrong
          </h2>
          <p className="text-red-600 mb-4">
            The application encountered an unexpected error. The issue has been logged for investigation.
          </p>
          <details className="text-xs text-gray-700 bg-white p-2 rounded border border-gray-200">
            <summary>Technical Details</summary>
            <pre className="mt-2 overflow-auto max-h-40">
              {this.state.error?.toString()}
            </pre>
          </details>
          <button
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 