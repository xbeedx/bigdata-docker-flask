const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");

var converter = new showdown.Converter({extensions: ['table']});

const loadDataFromLocalstorage = () => {
    const themeColor = localStorage.getItem("themeColor");
    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
    const defaultText = `<div class="default-text">
                            <img src="/static/images/chatgpt-logo.webp" width="90" height="90" alt="user-img"><br>
                            <h1>TuringGPT</h1>
                            <p>Comment puis-je vous aider aujourd'hui ?</p>
                        </div>`;
    chatContainer.innerHTML = localStorage.getItem("all-chats") || defaultText;
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
};

const createChatElement = (content, className) => {
    const chatDiv = document.createElement("div");
    chatDiv.classList.add("chat", className);
    chatDiv.innerHTML = content;
    return chatDiv;
};

const copyResponse = (copyBtn) => {
    const responseTextElement = copyBtn.parentElement.querySelector("p");
    navigator.clipboard.writeText(responseTextElement.textContent);
    copyBtn.textContent = "done";
    setTimeout(() => copyBtn.textContent = "content_copy", 1000);
};

const showTypingAnimation = () => {
    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="/static/images/chatgpt-logo.webp" width="90" height="90" alt="user-img">
                        <div class="typing-animation">
                            <div class="typing-dot" style="--delay: 0.2s"></div>
                            <div class="typing-dot" style="--delay: 0.3s"></div>
                            <div class="typing-dot" style="--delay: 0.4s"></div>
                        </div>
                    </div>
                </div>`;
    const incomingChatDiv = createChatElement(html, "incoming");
    chatContainer.appendChild(incomingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
};

const handleOutgoingChat = async () => {
    const userText = chatInput.value.trim();
    if(!userText) return;
    chatInput.value = "";

    const outgoingChatHtml = `<div class="chat-content">
                                <div class="chat-details">
                                    <img src="/static/images/tom1.PNG" alt="user-img">
                                    <p>${userText}</p>
                                </div>
                             </div>`;
    const outgoingChatDiv = createChatElement(outgoingChatHtml, "outgoing");
    chatContainer.appendChild(outgoingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);

    const defaultTextDiv = document.querySelector(".default-text");

    if (defaultTextDiv) {
        defaultTextDiv.style.display = 'none';
    }

    if (defaultTextDiv) {
        defaultTextDiv.remove();
    }

    showTypingAnimation();

    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userText })
    });

    chatContainer.removeChild(chatContainer.lastChild); 

    

    if (response.ok) {
        const data = await response.json();
      
        formattedContent = converter.makeHtml(data.bot);
        
        if(data.redirect)
        {
            window.location.href = '/'+data.redirect
            return
        }
        const incomingChatHtml = `<div class="chat-content">
                                    <div class="chat-details">
                                        <img src="/static/images/chatgpt-logo.webp" width="90" height="90" alt="user-img">
                                        <div class="markdown-body"
                                            <p>${formattedContent}</p>
                                        </div>
                                    </div>
                                    <span onclick="copyResponse(this)" class="material-symbols-rounded">content_copy</span>
                                  </div>`;
        const incomingChatDiv = createChatElement(incomingChatHtml, "incoming");
        console.log(incomingChatDiv)

        chatContainer.appendChild(incomingChatDiv);
        chatContainer.scrollTo(0, chatContainer.scrollHeight);
        hljs.highlightAll({});
    } else {
        const errorDiv = createChatElement("Erreur : impossible de recevoir une réponse.", "error");
        chatContainer.appendChild(errorDiv);
    }
};

sendButton.addEventListener("click", handleOutgoingChat);
chatInput.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        // Execute handleOutgoingChat when Ctrl + Enter are pressed
        handleOutgoingChat();
    }
});


deleteButton.addEventListener("click", () => {
    if(confirm("Voulez-vous vraiment supprimer toutes les conversations ?")) {
        localStorage.removeItem("all-chats");
        loadDataFromLocalstorage();
    }
});

themeButton.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
});

document.addEventListener('DOMContentLoaded', (event) => {
    const themeColor = localStorage.getItem("themeColor") || 'dark_mode';
    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = themeColor === "light_mode" ? "dark_mode" : "light_mode";
});


loadDataFromLocalstorage();
