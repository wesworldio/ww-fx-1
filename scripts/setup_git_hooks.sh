#!/bin/bash
# Setup git hooks to automatically regenerate build-info.json

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

HOOKS_DIR="$REPO_ROOT/.git/hooks"
PRE_COMMIT_HOOK="$HOOKS_DIR/pre-commit"
POST_COMMIT_HOOK="$HOOKS_DIR/post-commit"

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Create pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash
# Git pre-commit hook to regenerate build-info.json before commit
# This ensures build-info.json is up to date with the commit about to be made

REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_INFO_SCRIPT="$REPO_ROOT/scripts/generate_build_info.py"
BUILD_INFO_FILE="$REPO_ROOT/build-info.json"

# Check if the script exists
if [ ! -f "$BUILD_INFO_SCRIPT" ]; then
    exit 0
fi

# Regenerate build-info.json based on current HEAD (before new commit)
# This gives us the build number and commit info that will be in the commit
python3 "$BUILD_INFO_SCRIPT" > /dev/null 2>&1

# Stage the updated build-info.json if it exists
if [ -f "$BUILD_INFO_FILE" ]; then
    git add "$BUILD_INFO_FILE" 2>/dev/null || true
fi

exit 0
EOF

# Create post-commit hook
cat > "$POST_COMMIT_HOOK" << 'EOF'
#!/bin/bash
# Git post-commit hook to regenerate build-info.json after each commit
# This ensures build-info.json always reflects the latest commit

REPO_ROOT="$(git rev-parse --show-toplevel)"
BUILD_INFO_SCRIPT="$REPO_ROOT/scripts/generate_build_info.py"
BUILD_INFO_FILE="$REPO_ROOT/build-info.json"

# Check if the script exists
if [ ! -f "$BUILD_INFO_SCRIPT" ]; then
    exit 0
fi

# Regenerate build-info.json with the new commit info
python3 "$BUILD_INFO_SCRIPT" > /dev/null 2>&1

# Check if build-info.json was modified (it should be, since we just committed)
if ! git diff --quiet "$BUILD_INFO_FILE" 2>/dev/null; then
    # File was updated, stage it for the next commit
    git add "$BUILD_INFO_FILE" > /dev/null 2>&1
    echo "✓ build-info.json updated (will be included in next commit)"
fi

exit 0
EOF

# Make hooks executable
chmod +x "$PRE_COMMIT_HOOK"
chmod +x "$POST_COMMIT_HOOK"

echo "✓ Git hooks installed successfully"
echo "  - pre-commit: Regenerates build-info.json before each commit"
echo "  - post-commit: Regenerates build-info.json after each commit"
echo ""
echo "build-info.json will now be automatically updated on every commit"

