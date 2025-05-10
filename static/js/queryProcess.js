document.getElementById('chat-form').addEventListener('submit', async function(event) {
  event.preventDefault();
  const query = document.getElementById('query').value.trim();
  const responseDiv = document.getElementById('response');
  responseDiv.textContent = 'Loading...';
  
  try {
      const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query })
      });
      const data = await res.json();
      if (data.error) {
          responseDiv.textContent = 'Error: ' + data.error;
      } else {
          // Create a new AI chat bubble
          const aiBubble = document.createElement('div');
          aiBubble.classList.add('chat-bubble', 'ai');
          aiBubble.innerHTML = `<div class="message">${data.response}</div>`;
          responseDiv.appendChild(aiBubble);

          // Optionally, add user bubble
          const userBubble = document.createElement('div');
          userBubble.classList.add('chat-bubble', 'user');
          userBubble.innerHTML = `<div class="message">${query}</div>`;
          responseDiv.insertBefore(userBubble, responseDiv.firstChild);
      }
  } catch (error) {
      responseDiv.textContent = 'Error: ' + error;
  }
});
