import React, { useState } from 'react';
import './App.css';
import Modal from 'react-modal';

// Modal을 사용할 때 접근성을 위해 필수적으로 설정해야 합니다.
Modal.setAppElement('#root');

function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState('');
  const [modalIsOpen, setModalIsOpen] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, content }),  // 제목과 내용을 JSON으로 변환하여 전송
      });

      if (!response.ok) {
        throw new Error('네트워크 응답이 실패했습니다.');
      }

      const data = await response.json();
      setResult(data.result);  // 결과를 state에 저장
      setModalIsOpen(true);    // 결과를 받은 후 모달 열기
    } catch (error) {
      console.error('예측 중 오류 발생:', error);
      setResult('예측 실패');
      setModalIsOpen(true);    // 오류 발생 시에도 모달 열기
    }
  };

  const closeModal = () => {
    setModalIsOpen(false);     // 모달 닫기
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
          placeholder="메일 제목을 입력하세요"
        />
        <textarea
          className='content'
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="메일 내용을 입력하세요"
        />
        <button type="submit">스팸 검사</button>
      </form>

      {/* Modal 컴포넌트 */}
      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}   // 모달 외부 클릭 시 닫기
        contentLabel="결과 팝업"
        style={{
          content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
          },
        }}
      >
        <p>{result}인 것 같아요!!</p>
        <button onClick={closeModal}>확인</button>
      </Modal>
    </div>
  );
}

export default App;