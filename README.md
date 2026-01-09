# 🚀 Learning Guide

> Aiming to build a comprehensive, open-source learning resource for AI fundamemtals.

**🌐 Live Site**: [https://zhangjunwei0233.github.io/LearningGuide/](https://zhangjunwei0233.github.io/LearningGuide/)

## 📖 What is Learning Guide?

Learning Guide is a collaborative knowledge base designed to provide systematic learning paths and detailed explanations for AI newcomers. Built with [Quartz v4](https://quartz.jzhao.xyz/), it transforms Markdown content into an interconnected web of knowledge with powerful features like graph visualization, full-text search, and wiki-style linking.

### 🎯 Our Mission
- Provide **rigorous yet accessible** explanations of core AI concepts
- Create **systematic learning paths** from fundamentals to advanced topics
- Build an **interconnected knowledge graph** that shows relationships between concepts
- Maintain **academic quality** with proper mathematical formulations and proofs

## 📚 Current Coverage

### 🤖 Machine Learning
- **Supervised Learning**: Linear/Logistic Regression, SVM, Neural Networks
- **Unsupervised Learning**: K-Means Clustering, PCA
- **Model Training**: Cross-validation, learning curves, error analysis

## 🛠️ Development Setup

### Prerequisites
- **Node.js** v22+ 
- **npm** v10.9.2+
- **Git** for version control

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/LearningGuide.git
cd LearningGuide

# Install dependencies
npm install

# Start development server with hot reload
npx quartz build --serve

# Open http://localhost:8080
```

### Development Commands
```bash
# Build for production
npx quartz build

# Type check
npm run check

# Format code
npm run format

# Run tests
npm run test

# Serve documentation locally
npm run docs
```

## 🤝 How to Contribute

We welcome contributions from educators, students, and professionals! Here's how you can help:

### 📝 Contributing Content

#### 1. **Add New Topics**
- Follow the existing structure in `content/` directory
- Create both overview pages and detailed leaf nodes
- Use our [[wikilink]] system for cross-references

#### 2. **Improve Existing Content**
- Fix mathematical errors or unclear explanations
- Add practical examples and applications
- Enhance cross-references between topics

#### 3. **Add Visual Elements**
- Create diagrams and illustrations
- Add mathematical visualizations
- Improve the learning experience

### 📁 Project Structure
```
LearningGuide/
├── content/                      # All learning content
│   ├── index.md                  # Main landing page
│   ├── 机器学习/                 # Machine Learning section
│   │   ├── 机器学习总览.md       # ML overview (navigation page)
│   │   ├── 监督学习/             # Supervised learning (navigation)
│   │   └── ...
│   └── ...
├── quartz/                     # Quartz framework (don't edit)
├── public/                     # Generated site (auto-generated)
├── quartz.config.ts            # Site configuration
└── quartz.layout.ts            # Page layout configuration
```

### 🔄 Contribution Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b add-new-topic`
3. **Write your content**
4. **Test locally**: `npx quartz build --serve`
5. **Submit a pull request** with:
   - Clear description of what you added/changed
   - Screenshots if you added visual elements
   - Test confirmation that links work properly

## 🌍 Deployment

The site automatically deploys to Vercel on every push to the main branch. For manual deployment:

```bash
# Build the site
npx quartz build

# Deploy to your hosting platform
# (Vercel, Netlify, GitHub Pages, etc.)
```

## 🙏 Acknowledgments

- Built with [Quartz v4](https://quartz.jzhao.xyz/) by @jackyzha0
- Thanks to all contributors who help make learning accessible

---

**Ready to contribute?** Start by exploring our [content structure](https://learning-guide.vercel.app/), pick a topic you're passionate about, and help us build the best AI learning resource on the web! 🚀
