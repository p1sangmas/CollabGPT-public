#!/usr/bin/env python3
"""
Entry point script for CollabGPT.

This script provides command-line access to CollabGPT's functionality for real-world usage.
Usage examples:
  python main.py                                    # Start the CollabGPT service
  python main.py --action smart_suggestions --doc_id DOC_ID  # Generate smart edit suggestions
  python main.py --action prompt_chain --doc_id DOC_ID --chain_type analysis  # Run analysis chain
"""

import sys
import os
import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.app import CollabGPT
from src.utils import logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CollabGPT - AI Assistant for Collaborative Document Editing')
    
    # Basic arguments
    parser.add_argument('--doc_id', type=str, help='Google Document ID to process')
    parser.add_argument('--action', type=str, default='start',
                      choices=['start', 'process', 'smart_suggestions', 'context_window', 
                               'document_map', 'prompt_chain', 'daily_summary', 'monitor'],
                      help='Action to perform')
    
    # Action-specific arguments
    parser.add_argument('--section', type=str, help='Document section to focus on')
    parser.add_argument('--query', type=str, help='Query for context window or search')
    parser.add_argument('--chain_type', type=str, 
                      choices=['analysis', 'summary', 'suggestions'],
                      help='Type of prompt chain to execute')
    parser.add_argument('--max_suggestions', type=int, default=3, 
                      help='Maximum number of suggestions to generate')
    parser.add_argument('--include_history', action='store_true', 
                      help='Include document history in context')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--analyze', action='store_true', 
                      help='Perform deep analysis on the document')
    
    # System arguments
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Get a logger instance
    log = logger.get_logger("main")
    
    # If verbose mode, set the logger level to DEBUG
    if args.verbose:
        import logging
        logger.logger.setLevel(logging.DEBUG)
        log.debug("Debug logging enabled")
    
    log.info("Starting CollabGPT")
    
    # Create application instance
    app = CollabGPT()
    
    # Initialize application
    if not app.initialize():
        log.error("Failed to initialize CollabGPT")
        return 1
    
    # Explicitly authenticate with Google Docs API
    if not app.google_docs_api.authenticate():
        log.error("Failed to authenticate with Google Docs API")
        return 1
    log.info("Successfully authenticated with Google Docs API")
    
    # Start application services if in server mode
    if args.action == 'start':
        if not app.start():
            log.error("Failed to start CollabGPT services")
            return 1
            
        log.info("CollabGPT services started. Press Ctrl+C to exit.")
        try:
            # Keep the main thread alive
            while app.running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received, shutting down...")
        finally:
            if app.running:
                app.stop()
                
        return 0
    
    # Handle document-specific actions
    if args.doc_id:
        log.info(f"Processing document: {args.doc_id}")
        
        # Process the document (basic analysis)
        if args.action == 'process':
            results = app.process_document(args.doc_id)
            _output_results(results, args.output)
            
        # Generate smart edit suggestions
        elif args.action == 'smart_suggestions':
            suggestions = app.generate_smart_edit_suggestions(
                args.doc_id, 
                max_suggestions=args.max_suggestions,
                feedback_enabled=True
            )
            _output_results(suggestions, args.output)
            
        # Get context window
        elif args.action == 'context_window':
            context = app.get_context_window(
                args.doc_id,
                focus_section=args.section,
                query=args.query,
                include_history=args.include_history
            )
            _output_results(context, args.output)
            
        # Get document map
        elif args.action == 'document_map':
            doc_map = app.get_document_map(args.doc_id)
            _output_results(doc_map, args.output)
            
        # Run prompt chain
        elif args.action == 'prompt_chain':
            if not args.chain_type:
                log.error("Chain type (--chain_type) is required for prompt_chain action")
                return 1
                
            chain_results = app.run_prompt_chain(
                args.doc_id,
                chain_type=args.chain_type
            )
            _output_results(chain_results, args.output)
            
        # Generate daily summary
        elif args.action == 'daily_summary':
            # Get document activity
            activity = app.get_document_activity(args.doc_id, hours=24)
            
            # Get document analysis with the summary chain
            summary = app.run_prompt_chain(args.doc_id, chain_type='summary')
            
            # Combine results
            results = {
                'timestamp': datetime.now().isoformat(),
                'document_id': args.doc_id,
                'activity': activity,
                'summary': summary
            }
            _output_results(results, args.output or f"daily_summary_{args.doc_id}_{datetime.now().strftime('%Y%m%d')}.json")
            
        # Monitor document (similar to monitor_doc.py)
        elif args.action == 'monitor':
            log.info(f"Starting to monitor document: {args.doc_id}")
            
            if args.analyze:
                # First do a full analysis of the document
                analysis = app.process_document(args.doc_id)
                log.info(f"Initial document analysis complete. Found {len(analysis.get('sections', []))} sections.")
                
            # Start the application services for monitoring
            if not app.start():
                log.error("Failed to start CollabGPT services for monitoring")
                return 1
                
            # Set up document webhook if not already monitored
            if args.doc_id not in app.monitored_documents:
                app._setup_document_webhook(args.doc_id)
                
            log.info(f"Now monitoring document {args.doc_id}. Press Ctrl+C to stop.")
            try:
                # Keep the main thread alive
                while app.running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                log.info("Keyboard interrupt received, shutting down...")
            finally:
                if app.running:
                    app.stop()
        
        else:
            log.error(f"Unknown action for document: {args.action}")
            return 1
    else:
        if args.action != 'start':
            log.error("Document ID (--doc_id) is required for this action")
            return 1
    
    return 0

def _output_results(results, output_path=None):
    """Output results to stdout or a file."""
    log = logger.get_logger("main")
    
    if output_path:
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write to file
        with open(output_path, 'w') as f:
            if isinstance(results, dict) or isinstance(results, list):
                json.dump(results, f, indent=2)
            else:
                f.write(str(results))
        log.info(f"Results written to {output_path}")
    else:
        # Print to stdout
        if isinstance(results, dict) or isinstance(results, list):
            print(json.dumps(results, indent=2))
        else:
            print(results)

if __name__ == "__main__":
    sys.exit(main())