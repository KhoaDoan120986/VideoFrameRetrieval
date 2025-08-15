import React, { useState, useEffect } from "react";
import { FaArrowUp, FaSearch } from "react-icons/fa";
import { FiSettings } from "react-icons/fi";
import "./App.css";

const resizeObserverErrHandler = e => {
  if (e.message === "ResizeObserver loop completed with undelivered notifications.") {
    return;
  }
  console.error(e);
};
window.addEventListener("error", resizeObserverErrHandler);

function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [pageIndex, setPageIndex] = useState(0);
  const pageSize = 8;
  
  async function fetchSearchResults(query) {
    try {
      const response = await fetch("http://127.0.0.1:8000/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          topK: topK
        }),
      });
      if (!response.ok) throw new Error("Failed to fetch");
      const data = await response.json();
      console.log(data.results)
      return data.results;
    } catch (error) {
      console.error(error);
      return [];
    }
  }

  async function search() {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const data = await fetchSearchResults(query);
    console.log(data);
    const mappedResults = data.map(item => ({
      ...item,
      image: `/frames/${item.image}/frame_0000.jpg`
    }));
    setResults(mappedResults);
  }

  useEffect(() => {
    document.title = "AIC";
  }, []);

  const totalPages = Math.max(Math.ceil(results.length / pageSize), 1);

  const currentPageData = results.slice(
    pageIndex * pageSize,
    (pageIndex + 1) * pageSize
  );


  const goToPage = (newPageIndex) => {
    if (newPageIndex < 0) newPageIndex = 0;
    else if (newPageIndex >= totalPages) newPageIndex = totalPages - 1;
    setPageIndex(newPageIndex);
  };
  const missingCount = pageSize - currentPageData.length;

  return (
      <div className="container">
        <div className="sidebar">
          <div className="logo">AI CHALLENGE 2025</div>
          <div className="team-info">
            <h2> GROUP: Tralalero Tralala </h2>
            <ul>
              <li>Trần Nguyên Huân</li>
              <li>Trần Hải Phát</li>
              <li>Nguyễn Bảo Tuấn</li>
              <li>Nguyễn Phát Đạt</li>
              <li>Doãn Anh Khoa</li>
            </ul>
          </div>
          <div className="settings-title"> <FiSettings /> MODE </div>
          <div className="sidebar-content">
             <label className="topk-label" >
              Top K: <input className="topk-input" type="number" min="10" max="100" step="1" 
              value={topK} onChange={(e) => setTopK(Number(e.target.value))}/>
            </label>
          </div>
        </div>

        <div className="main-content">
          {results.length === 0 ? (
            <div className="empty-state" style={{ textAlign: "center", marginTop: 50 }}>
              <FaSearch size={80} color="#999" />
              <p>Chưa có kết quả để hiển thị</p>
            </div>
          ) : (
            <>
              <div className="result-box">
                {currentPageData.map((item, idx) => (
                  <div key={idx} className="card">
                    <img src={item.image} alt=""/>
                    <p>{pageIndex * pageSize + idx + 1}. {item.caption}</p>
                  </div>
                ))}
                {Array.from({ length: missingCount }).map((_, idx) => (
                  <div key={"empty-" + idx} className="card-empty"></div>
                ))}
              </div>
              <div className="pagination">
                <button onClick={() => goToPage(pageIndex - 1)} disabled={pageIndex === 0}>
                  Previous
                </button>

                <span>
                  Page {pageIndex + 1} of {totalPages}
                </span>

                <button
                  onClick={() => goToPage(pageIndex + 1)}
                  disabled={pageIndex === totalPages - 1}
                >
                  Next
                </button>
              </div>
            </>
          )}

          <div className="search-row">
            <input
              type="text"
              placeholder="Enter text query..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  search();
                }
              }}
            />
            <button id="search-btn" onClick={search}>
              <FaArrowUp />
            </button>
          </div>
        </div>
      </div>
  );
}

export default App;
