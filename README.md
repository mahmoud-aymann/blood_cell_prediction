# 🩸 Blood Cell Diagnostics
##
> **Complete End-to-End AI Solution**: From raw blood cell images to production-ready web application with 95%+ accuracy

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-brightgreen.svg)](https://web-production-4a424.up.railway.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/mahmoud-aymann/blood_cell_prediction)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Complete End-to-End Pipeline

This project demonstrates mastery of the entire machine learning lifecycle:

### 🔬 **Data Collection & Preprocessing**
- Curated 17,000+ blood cell images across 8 categories
- Professional data preprocessing and augmentation

### 🧠 **Model Development**
- Built custom CNN architecture from scratch
- Implemented proper train/test splits with cross-validation
- Achieved 95%+ accuracy through hyperparameter tuning

### 🚀 **Production Deployment**
- Developed full-stack Flask application with modern UI
- Containerized with Docker and deployed on Railway
- Real-time inference under 50ms with privacy-first design

### 🎯 **Key Features**
- **8 Blood Cell Types**: Classifies basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil, and platelet
- **Flask RESTful API**: Seamless image upload and processing endpoints
- **Real-time AI Analysis**: Lightning-fast inference with confidence scores
- **Professional UI**: Modern, responsive interface with drag-and-drop upload
- **Privacy-First**: No image storage; processing happens in-memory
- **Production-Ready**: Robust error handling and user feedback

## 🚀 Live Demo

**🔗 [Try the Live Application](https://web-production-4a424.up.railway.app/)**

Upload any blood smear image and get instant AI-powered classification results!

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

## 🛠️ Complete Tech Stack (End-to-End)

### **Data Processing**
- **Python**: Core programming language
- **OpenCV**: Image processing and manipulation
- **NumPy & Pandas**: Data manipulation and analysis

### **AI/ML Pipeline**
- **PyTorch**: Deep learning framework
- **Custom CNN**: 3-layer convolutional neural network
- **Computer Vision**: Image classification and preprocessing

### **Backend Framework**
- **Flask**: Python web framework
- **RESTful APIs**: Image upload and processing endpoints
- **Request Handling**: Secure file upload and session management

### **Frontend Development**
- **HTML5**: Semantic markup
- **CSS3**: Modern styling and responsive design
- **JavaScript**: Interactive user interface and AJAX

### **DevOps & Deployment**
- **Docker**: Containerization
- **Railway**: Cloud hosting platform
- **Git**: Version control

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/mahmoud-aymann/blood_cell_prediction.git
cd blood_cell_prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Model

Place your trained model file as `cnn_model.pth` in the project root, or set the `MODEL_PATH` environment variable.

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `cnn_model.pth` | Path to the PyTorch model file |
| `IMG_NORMALIZE` | `false` | Enable image normalization |
| `IMG_MEAN` | `0.485,0.456,0.406` | Normalization mean values |
| `IMG_STD` | `0.229,0.224,0.225` | Normalization std values |
| `PORT` | `5000` | Server port |
| `FLASK_SECRET_KEY` | `dev-secret-key` | Flask secret key |

### Model Requirements

- **Input Size**: 128×128 RGB images
- **Format**: PyTorch state_dict or TorchScript model
- **Classes**: 8 blood cell types (see CLASS_LABELS in app.py)

## 🚀 Deployment on Railway

### Method 1: One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

### Method 2: Manual Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Railway**:
   - Go to [Railway](https://railway.app)
   - Sign in with GitHub
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your forked repository

3. **Configure Environment Variables**:
   ```
   MODEL_PATH=cnn_model.pth
   IMG_NORMALIZE=false
   PORT=5000
   ```

4. **Upload Model File**:
   - Add your `cnn_model.pth` file to the project root
   - Commit and push to trigger deployment

5. **Deploy**:
   - Railway will automatically build and deploy your application
   - Your app will be available at the provided Railway URL

### Method 3: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

## 📁 Project Structure

```
blood_cell_prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── cnn_model.pth         # Trained PyTorch model
├── blood.ipynb           # Jupyter notebook for model training
├── templates/
│   ├── index.html        # Upload interface
│   ├── result.html       # Results display
│   └── about.html        # About page
├── static/
│   ├── styles.css        # Styling
│   ├── ui.js            # Upload interactions
│   ├── result.js        # Results interactions
│   └── logo.svg         # Brand logo
├── bloodcells_dataset/   # Training dataset
│   ├── train/           # Training images
│   └── test/            # Test images
├── Dockerfile           # Docker configuration
├── Procfile            # Railway deployment config
├── railway.json        # Railway configuration
└── README.md
```

## 🧠 Model Architecture

The CNN model consists of:
- **3 Convolutional Layers**: 3×3 kernels with ReLU activation
- **MaxPool2d**: 2×2 pooling after each conv layer
- **Dropout**: 0.3 dropout for regularization
- **Fully Connected**: 256 → 8 classes

```
Input: [1, 3, 128, 128]
Conv1 + Pool: [1, 32, 64, 64]
Conv2 + Pool: [1, 64, 32, 32]  
Conv3 + Pool: [1, 128, 16, 16]
Flatten: [1, 32768]
FC1: [1, 256]
FC2: [1, 8]
```

## 🔬 Supported Cell Types

| Class | Description |
|-------|-------------|
| Basophil | White blood cell with large granules |
| Eosinophil | White blood cell with red-orange granules |
| Erythroblast | Immature red blood cell |
| Immature Granulocyte | Young white blood cell |
| Lymphocyte | Small white blood cell |
| Monocyte | Large white blood cell |
| Neutrophil | Most common white blood cell |
| Platelet | Blood clotting cell |

## 📊 Performance Metrics

- **Inference Time**: ~10-40ms on CPU (typical)
- **Model Accuracy**: 95%+ on test dataset
- **Input Size**: 128×128 RGB images
- **Max File Size**: 10MB per upload
- **Model Size**: Lightweight CNN (3 conv blocks)
- **Latency**: Sub-50ms real-time processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Educational Background

This project was developed as part of the **Advanced AI & Machine Learning Diploma** program at **[AMIT LEARNING](https://amitlearning.com)**, under the expert guidance of **George Samuel**. The comprehensive curriculum covering computer vision, deep learning, and practical AI applications provided the perfect foundation for creating this medical diagnostics system.

### Course Highlights:
- **Institution**: AMIT LEARNING
- **Instructor**: George Samuel
- **Program**: Advanced AI & Machine Learning Diploma
- **Specialization**: Computer Vision & Deep Learning
- **Focus**: Practical applications of AI in healthcare

## 👨‍💻 Author

**Mahmoud Ayman**
- 🎓 Communications & Electronics Engineering Student
- 🤖 AI/ML & Computer Vision Specialist  
- 🌍 Based in Egypt
- 📧 Email: [mahmoudaymann153@gmail.com](mailto:mahmoudaymann153@gmail.com)
- 💼 LinkedIn: [in/mahmoud-aymann](https://linkedin.com/in/mahmoud-aymann)
- 📝 Medium: [@mahmoudayman1](https://medium.com/@mahmoudayman1)
- 🐙 GitHub: [@mahmoud-aymann](https://github.com/mahmoud-aymann)

## 🙏 Acknowledgments

- **AMIT LEARNING** - Educational institution providing the Advanced AI & Machine Learning Diploma program
- **George Samuel** - Expert instructor and mentor for computer vision and deep learning concepts
- PyTorch team for the deep learning framework
- Flask team for the web framework
- Railway for hosting platform
- Medical imaging community for datasets and research

## 📞 Support

If you have any questions or run into issues:

1. Check the [Issues](https://github.com/mahmoud-aymann/blood-cell-diagnostics/issues) page
2. Create a new issue with detailed information
3. Contact: [mahmoudaymann153@gmail.com](mailto:mahmoudaymann153@gmail.com)

---

⭐ **Star this repository if you found it helpful!**

## 🔗 Quick Links

- **Live Demo**: [https://web-production-4a424.up.railway.app/](https://web-production-4a424.up.railway.app/)
- **GitHub Repository**: [https://github.com/mahmoud-aymann/blood_cell_prediction](https://github.com/mahmoud-aymann/blood_cell_prediction)
- **Test Images**: [Google Drive - Sample Blood Cell Images](https://drive.google.com/drive/folders/1wxwsKxpJz-O5fHqS5ROGDJ4tG6Ac5Hjq?usp=sharing)
---

**This end-to-end project showcases complete mastery of the machine learning pipeline, from data collection to production deployment! 🚀**

#













