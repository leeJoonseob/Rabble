import React, { useState } from 'react';
import './App.css';
import Modal from 'react-modal';

Modal.setAppElement('#root');

function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState('');
  const [modalIsOpen, setModalIsOpen] = useState(false);

  const handlePredict = async (method) => {
    try {
      const response = await fetch(`http://127.0.0.1:8080/predict_${method}`, {
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
      setResult(`${data.result}\n`);
      setModalIsOpen(true);
    } catch (error) {
      console.error(`${method} 예측 중 오류 발생:`, error);
      setResult(`${method.toUpperCase()} 예측 실패`);
      setModalIsOpen(true);
    }
  };

  const closeModal = () => {
    setModalIsOpen(false);
  };

  return (
    <div className="App">
      <div className="header">
        <img src="/spam_guardian-removebg-preview.png" alt="Spam Guardian Logo" />
        <h1>DCCD</h1>
      </div>
      
      <form onSubmit={(e) => e.preventDefault()}>
        <textarea
          className='title'
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="메일 제목을 입력하세요"
        />
        <textarea
          className='content'
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="메일 내용을 입력하세요"
        />
        <div className="button-container">
          <button onClick={() => handlePredict('deeplearning')}>딥러닝 스팸 검사</button>
          <button onClick={() => handlePredict('machinelearning')}>머신러닝 스팸 검사</button>
        </div>
      </form>

      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        contentLabel="결과 팝업"
        style={{
          content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '200px',
            height: '100px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '20px',
          },
        }}
      >
        <p>{result}</p>
        <button onClick={closeModal}>확인</button>
      </Modal>
    </div>
  );
}

export default App;