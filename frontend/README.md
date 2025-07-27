# Fashion Search Frontend

A modern, elegant React frontend for the Amazon Fashion Search API. This application provides an intuitive interface for semantic product search powered by OpenAI embeddings.

## Features

- 🔍 **Semantic Search**: Natural language product search
- 🎨 **Modern UI**: Glassmorphism design with smooth animations
- 📱 **Responsive**: Works on desktop, tablet, and mobile
- ⚡ **Real-time**: Live API health monitoring
- 🔧 **Advanced Filters**: Price, rating, and category filters
- 📊 **Rich Results**: Product cards with images, ratings, and features

## Quick Start

### Prerequisites

- Node.js 16+ 
- npm or yarn
- Fashion Search API running on `http://localhost:8000`

### Installation

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

3. **Open your browser** to `http://localhost:3000`

### Production Build

```bash
npm run build
npm install -g serve
serve -s build
```

## Configuration

### API URL

The frontend expects the API to be running on `http://localhost:8000` by default. To change this:

1. **Development**: The `package.json` includes a proxy configuration
2. **Production**: Set the `REACT_APP_API_URL` environment variable:
   ```bash
   REACT_APP_API_URL=https://your-api-domain.com npm run build
   ```

## Usage

### Basic Search

1. Type a natural language query in the search bar
2. Press Enter or click the search button
3. Browse the results in the grid layout

**Example queries**:
- "comfortable summer dresses under $50"
- "elegant wedding guest outfit"
- "professional work attire"

### Advanced Filters

1. Click the filter icon in the search bar
2. Set price ranges, minimum ratings, or result count
3. Apply filters to refine your search

### Features

- **Live Status**: API health indicator at the top
- **Sample Queries**: Click example queries to try them
- **Similarity Scores**: See how well each product matches your query
- **Product Details**: View prices, ratings, features, and categories

## Architecture

```
frontend/
├── public/          # Static assets
├── src/
│   ├── components/  # React components
│   │   ├── Header.js/css
│   │   ├── SearchBar.js/css
│   │   ├── ProductResults.js/css
│   │   └── StatusIndicator.js/css
│   ├── services/    # API communication
│   │   └── api.js
│   ├── App.js/css   # Main application
│   ├── index.js/css # Entry point
│   └── ...
└── package.json     # Dependencies
```

## Technologies

- **React 18**: Frontend framework
- **Axios**: HTTP client
- **Lucide React**: Icon library
- **CSS Variables**: Design system
- **Glassmorphism**: Modern visual design

## Design System

The app uses CSS custom properties for consistent theming:

- **Colors**: Primary blues and purples with glassmorphism
- **Typography**: Inter font family
- **Spacing**: Consistent spacing scale
- **Shadows**: Layered depth effects
- **Animations**: Smooth transitions and micro-interactions

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Development

### Available Scripts

- `npm start`: Development server with hot reload
- `npm run build`: Production build
- `npm test`: Run tests
- `npm run eject`: Eject from Create React App

### Code Style

- ES6+ JavaScript
- Functional components with hooks
- CSS modules with BEM naming
- Responsive-first design

## Troubleshooting

### API Connection Issues

1. **Check API Status**: The status indicator shows API health
2. **CORS Issues**: The API includes CORS middleware
3. **Network**: Ensure API is running on the correct port

### Performance

- Images are lazy-loaded and optimized
- Search results are paginated (12 items default)
- Debounced API calls prevent excessive requests

## License

This project is part of the Amazon Fashion Search take-home assessment.