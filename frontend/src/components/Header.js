import React from 'react';
import { Search } from 'lucide-react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <Search className="logo-icon" size={32} />
            <div className="logo-text">
              <h1 className="logo-title">Fashion Search</h1>
              <p className="logo-subtitle">AI-Powered Product Discovery</p>
            </div>
          </div>
          
          <div className="header-info">
            <div className="info-badge">
              <span className="info-label">Powered by</span>
              <span className="info-value">OpenAI Embeddings</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;