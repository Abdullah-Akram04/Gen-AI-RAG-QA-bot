import { useState } from "react";
import Message from "./Message";

function Chat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const sendQuestion = async () => {
    if (!question.trim()) return;

    const userMessage = {
      role: "user",
      content: question
    };

    setMessages((previous) => [...previous, userMessage]);
    setQuestion("");

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: question
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: "assistant",
        content: data.answer
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage
      ]);
    } catch (error) {
      const errorMessage = {
        role: "assistant",
        content: "Unable to connect to the backend."
      };

      setMessages((previous) => [
        ...previous,
        errorMessage
      ]);
    }
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((message, index) => (
          <Message
            key={index}
            role={message.role}
            content={message.content}
          />
        ))}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={question}
          placeholder="Ask a question..."
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              sendQuestion();
            }
          }}
        />

        <button onClick={sendQuestion}>
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat;