function Message({ role, content }) {
  return (
    <div className={`message ${role}`}>
      <div className="message-role">
        {role === "user" ? "You" : "Assistant"}
      </div>
      <div className="message-content">
        {content}
      </div>
    </div>
  );
}

export default Message;