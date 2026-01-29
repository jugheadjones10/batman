# Batman Documentation

This directory contains the complete documentation for Batman, built with MkDocs and the shadcn theme.

## Viewing the Documentation

### Local Development

To view the documentation locally:

```bash
# Install dependencies
pip install -r requirements-docs.txt

# Or with uv
uv pip install mkdocs mkdocs-shadcn pymdown-extensions

# Serve the documentation
mkdocs serve
# or
uv run mkdocs serve
```

Then open http://localhost:8000 in your browser.

### Build Static Site

To build the static HTML site:

```bash
mkdocs build
# or
uv run mkdocs build
```

The built site will be in the `site/` directory.

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

## Documentation Structure

```
docs/
├── index.md                 # Home page
├── getting-started.md       # Installation and quick start
├── cli/                     # CLI tool documentation
│   ├── index.md            # CLI overview
│   ├── train.md            # Training tool
│   ├── inference.md        # Inference tool
│   ├── benchmark-latency.md
│   ├── compare-latency.md
│   ├── create-latency-video.md
│   ├── create-sidebyside-video.md
│   ├── importer.md
│   └── classes.md
├── scripts/                 # SLURM script documentation
│   ├── index.md            # Scripts overview
│   ├── submit-train.md
│   ├── submit-inference.md
│   ├── submit-benchmark.md
│   └── run-dev.md
├── guides/                  # Workflow guides
│   ├── training.md
│   ├── inference.md
│   ├── benchmarking.md
│   └── slurm.md
├── api/                     # API reference
│   └── index.md
├── javascripts/             # Interactive features
│   └── command-builder.js  # Command builder widget
└── stylesheets/            # Custom styles
    └── extra.css
```

## Interactive Command Builders

Many CLI tool and script pages include interactive command builders that:

- Show all available options with descriptions
- Generate commands as you select options
- Provide copy-to-clipboard functionality
- Save your preferences in localStorage

Look for the "Command Builder" sections on documentation pages.

## Writing Documentation

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add it to the `nav` section in `mkdocs.yml`
3. Follow the existing page structure and formatting

### Command Builder Integration

To add a command builder to a page:

```markdown
<div class="command-builder-widget" data-tool="tool_name" data-params='[
  {"name": "param1", "type": "text", "required": true, "description": "..."},
  {"name": "param2", "type": "number", "default": 50, "description": "..."}
]'></div>
```

Parameter types:

- `text` - Text input
- `number` - Number input
- `choice` - Dropdown select
- `flag` - Checkbox
- `path` - File/directory path

### Code Blocks

Use triple backticks with language identifier:

\`\`\`bash
./submit_train.sh --project data/projects/MyProject
\`\`\`

### Admonitions

Use for notes, warnings, and tips:

```markdown
!!! note
This is a note

!!! warning
This is a warning

!!! tip
This is a tip
```

## Theme

The documentation uses the [mkdocs-shadcn](https://asiffer.github.io/mkdocs-shadcn/) theme, which provides:

- Modern, clean design inspired by shadcn/ui
- Dark/light mode toggle
- Responsive layout
- Beautiful code syntax highlighting
- Full-text search

## Configuration

Main configuration is in `mkdocs.yml` at the project root.

Key settings:

- `site_name`: Site title
- `theme`: Theme configuration
- `nav`: Navigation structure
- `markdown_extensions`: Markdown features
- `extra_javascript`: Custom JavaScript
- `extra_css`: Custom CSS

## Maintenance

When adding new CLI tools or scripts:

1. Document the tool in the appropriate section
2. Add examples and use cases
3. Include an interactive command builder
4. Update the index/overview pages
5. Cross-link related pages
6. Test the documentation build

## Support

For documentation issues or suggestions:

- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review the [MkDocs documentation](https://www.mkdocs.org/)
