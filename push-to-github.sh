#!/bin/bash
# Script to push llama-jni-bridge to GitHub

echo "🚀 Pushing llama-jni-bridge to GitHub..."
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Error: Not in the project directory"
    exit 1
fi

# Check current status
echo "📊 Current Git Status:"
git status --short

echo ""
echo "📝 Recent commits:"
git log --oneline -5

echo ""
echo "🔗 Setting up GitHub remote..."
echo ""
echo "Please enter your GitHub repository URL"
echo "Example: https://github.com/uShure/llama-jni-bridge.git"
echo -n "Repository URL: "
read REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ No URL provided"
    exit 1
fi

# Add remote
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

echo ""
echo "📤 Pushing to GitHub..."
echo "You may be prompted for your GitHub credentials"
echo ""

# Try to push
if git push -u origin master; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🎉 Your repository is now available at: $REPO_URL"
else
    echo ""
    echo "❌ Push failed. Common solutions:"
    echo ""
    echo "1. If using HTTPS, you may need a Personal Access Token:"
    echo "   - Go to GitHub Settings → Developer settings → Personal access tokens"
    echo "   - Generate a token with 'repo' permissions"
    echo "   - Use the token as your password when prompted"
    echo ""
    echo "2. If you prefer SSH:"
    echo "   git remote set-url origin git@github.com:uShure/llama-jni-bridge.git"
    echo ""
    echo "3. If the repository doesn't exist yet:"
    echo "   - Create it at https://github.com/new"
    echo "   - Don't initialize with README/license"
    echo "   - Then run this script again"
fi
