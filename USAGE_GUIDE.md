# Real-World Usage Guide for CollabGPT

This guide provides instructions for setting up and using CollabGPT with your Google Docs for real-time collaboration assistance.

## Initial Setup

### 1. Set Up Google API Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the Google Docs API and Google Drive API
4. Create OAuth 2.0 credentials
5. Download the credentials JSON file
6. Save it to `credentials/google_credentials.json`

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources

```bash
python download_nltk_resources.py
```

### 4. Set Up Webhook (for real-time monitoring)

You'll need a publicly accessible URL for webhooks. You can use ngrok for testing:

```bash
# Install ngrok if not already installed
# Start ngrok on the port specified in your config (default: 8000)
ngrok http 8000
```

Then update your `.env` file with the ngrok URL:

```
WEBHOOK_EXTERNAL_URL=https://your-ngrok-url.ngrok-free.app
```

Finally, set up the webhook for a specific document:

```bash
# Set up webhook monitoring for a document
python setup_webhook.py YOUR_GOOGLE_DOC_ID
```

## Basic Usage

### 1. Start CollabGPT

```bash
python main.py
```

### 2. Monitor a Document

```bash
python main.py --action monitor --doc_id YOUR_GOOGLE_DOC_ID
```

Replace `YOUR_GOOGLE_DOC_ID` with the ID from your Google Doc URL:
https://docs.google.com/document/d/YOUR_GOOGLE_DOC_ID/edit

## Real-World Use Cases

### 1. Collaborative Document Writing

When multiple team members are working on the same document:

- CollabGPT monitors changes in real-time
- Detects potential conflicts when different people edit the same sections
- Provides summaries of changes made by team members
- Creates intelligent edit suggestions to improve document quality

### 2. Document Improvement

For existing documents that need improvement:

```bash
# Connect to an existing document
python monitor_doc.py --doc_id YOUR_GOOGLE_DOC_ID --analyze
```

CollabGPT will:
- Analyze the document structure
- Identify areas for improvement
- Generate specific edit suggestions with reasoning
- Provide a comprehensive document analysis

### 3. Writing Assistant

For ongoing document creation:

- CollabGPT observes your writing patterns
- Suggests improvements for clarity and consistency
- Helps maintain consistent style throughout the document
- Provides context-aware suggestions based on document history

### 4. Meeting Documentation

After team meetings:

- Add meeting notes to your Google Doc
- CollabGPT can help organize and structure the information
- Generate summaries and action items
- Link new content to relevant existing document sections

## Advanced Usage

### Using Context Windows

Context windows help the AI focus on specific parts of long documents:

```bash
# Get a focused context window for a specific section
python main.py --action context_window --doc_id YOUR_DOC_ID --section "Introduction"
```

### Running Prompt Chains

For sophisticated document analysis:

```bash
# Run an analysis prompt chain
python main.py --action prompt_chain --doc_id YOUR_DOC_ID --chain_type analysis
```

Available chain types: `analysis`, `summary`, `suggestions`

### Smart Edit Suggestions

To get intelligent edit suggestions with reasoning:

```bash
# Generate smart edit suggestions
python main.py --action smart_suggestions --doc_id YOUR_DOC_ID --max_suggestions 5
```

### Document Mapping

For understanding document structure:

```bash
# Generate a document map
python main.py --action document_map --doc_id YOUR_DOC_ID
```

## Integration with Team Workflows

### Example: Daily Document Reviews

1. Set up a scheduled task to run at the end of each day:

```bash
# Add to your crontab or scheduled tasks
python main.py --action daily_summary --doc_id YOUR_DOC_ID
```

2. CollabGPT will generate:
   - Summary of all changes made that day
   - List of active contributors
   - Potential areas that need attention
   - Suggestions for next steps

### Example: Content Approval Workflow

1. Writers add "@collabgpt review" comments when a section is ready for review
2. CollabGPT analyzes the section and provides improvement suggestions
3. Editors review both the content and AI suggestions
4. Final approval or further revision

## Practical Tips

1. **Start with smaller documents** until you're comfortable with the system
2. **Check the logs** (`data/collabgpt.log`) if you encounter issues
3. **Provide feedback** on suggestions to improve future recommendations
4. **Set up regular backups** of your documents when using experimental features

## Limitations

- Real-time monitoring requires a stable internet connection
- Very large documents (>100 pages) may experience slower processing
- The system works best with well-structured documents with clear headings
- Response time varies based on document size and complexity

## Troubleshooting

If you encounter issues:

1. Check the logs in `data/collabgpt.log`
2. Verify your Google API credentials are valid and have the correct permissions
3. Ensure your webhook URL is publicly accessible if using real-time features
4. For large documents, try using the section-focused features instead of whole document analysis

## Verifying Your Setup Is Working

When running CollabGPT, especially for document monitoring, you'll want to verify that everything is working correctly:

### 1. Check the Log File

Monitor the log file in real-time to see activity:

```bash
# View logs in real-time
tail -f data/collabgpt.log
```

Look for these indicators of successful operation:
- "Starting CollabGPT"
- "Initializing CollabGPT application"
- "Successfully authenticated with Google Docs API"
- "Webhook server started on port 8000"
- "Webhook set up for document: [YOUR_DOC_ID]"
- "Now monitoring document [YOUR_DOC_ID]"

### 2. Test Document Monitoring

To verify monitoring is working:
1. Start the monitoring script: `python main.py --action monitor --doc_id YOUR_DOC_ID`
2. Make a small change to the document in Google Docs
3. Check the logs for:
   - "Analyzing changes for document: [YOUR_DOC_ID]"
   - "Recorded edit [edit_id] by user [username]"
   - "Updated chunk section_X in document [YOUR_DOC_ID]"

### 3. Test Conflict Detection

To verify conflict detection:
1. Have two different users edit the same section within a short time window (default: 60s)
2. Check the logs for:
   - "Detected X conflicts for document [YOUR_DOC_ID]"
   - "Possible sequential conflict: [user1] and [user2] edited from the same base content"

### 4. Check RAG System Updates

After document changes, verify the RAG (Retrieval-Augmented Generation) system is updating:
1. Look for log entries like:
   - "Updated chunk section_X in document [YOUR_DOC_ID], now at version Y"
   - "Processed document [YOUR_DOC_ID]: created/updated Z chunks"

### 5. Verify LLM Integration

To confirm the LLM (Large Language Model) is responding:
1. Make document changes that should trigger LLM analysis
2. Check for log entries:
   - "Got successful response from OpenRouter (elapsed: X.XXs)"
   - "Executing step '[step_name]' with inputs: [...]"

### 6. Webhook Verification

To verify webhooks are functioning:
1. Ensure you see "External webhook URL: https://[...].ngrok-free.app/webhook" in logs
2. Make changes in Google Docs
3. Check logs for webhook events being received and processed

If any of these verification steps fail, refer to the Troubleshooting section above for resolution steps.