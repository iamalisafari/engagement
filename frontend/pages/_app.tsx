import '../styles/globals.css';
import type { AppProps } from 'next/app';
import ErrorBoundary from '../components/ErrorBoundary';
import { useEffect } from 'react';
import debugService from '../utils/debug-service';

function MyApp({ Component, pageProps }: AppProps) {
  useEffect(() => {
    // Set up global error handler for unhandled errors
    const handleGlobalError = (event: ErrorEvent) => {
      event.preventDefault();
      debugService.logClientError(event.error || new Error(event.message), {
        source: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        type: 'unhandled_error'
      }).catch(console.error);
    };

    // Set up global promise rejection handler
    const handleRejection = (event: PromiseRejectionEvent) => {
      event.preventDefault();
      const error = typeof event.reason === 'string' 
        ? new Error(event.reason)
        : event.reason;
      
      debugService.logClientError(error, {
        type: 'unhandled_rejection'
      }).catch(console.error);
    };

    // Add event listeners
    window.addEventListener('error', handleGlobalError);
    window.addEventListener('unhandledrejection', handleRejection);

    // Clean up on unmount
    return () => {
      window.removeEventListener('error', handleGlobalError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  }, []);

  return (
    <ErrorBoundary>
      <Component {...pageProps} />
    </ErrorBoundary>
  );
}

export default MyApp; 