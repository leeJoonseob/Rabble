import React, { useState } from 'react';
import './App.css';

function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, content }),
      });

      if (!response.ok) {
        throw new Error('네트워크 응답이 실패했습니다.');
      }  

      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      console.error('예측 중 오류 발생:', error);
      setResult('예측 실패',error);
    }
  };

  return (
    <div className="App">
      <div className="header">
        <img src="/spam_guardian-removebg-preview.png" alt="Spam Guardian Logo" />
        <h1>Spam Guardian</h1>
      </div>
      <form onSubmit={handleSubmit}>
        <textarea
          className='title'
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="메일제목을 입력하세요"
        />
        <textarea
          className='content'
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="메일내용을 입력하세요"
        />
        <button type="submit">스팸 검사</button>
      </form>
      {result && <p>결과: {result}</p>}
    </div>
  );
}

export default App;