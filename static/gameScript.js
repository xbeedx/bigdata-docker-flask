const guessInput = document.querySelector("#input-guess");
const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#button-send");
const chatResponse = document.querySelector("#answer-div");
const guessButton = document.querySelector("#button-guess");
const popup = document.querySelector("#popup");
const closePopup = document.querySelector("#close-popup");
const nbRequests = document.querySelector("#nbRequests");
const level = document.querySelector("#level");
const snackbar = document.getElementById('snackbar');

const createChatElement = (content, className) => {
    const chatDiv = document.createElement("div");
    chatDiv.classList.add("chat", className);

    let formattedContent = "";

    // Recherche des blocs de code
    if (content.includes("```")) {
        // Sépare le texte en blocs normaux et en blocs de code
        const parts = content.split("```");
        let isCodeBlock = false;

        parts.forEach(part => {
            if (isCodeBlock) {
                formattedContent += part;
            } else {
                formattedContent += part;
            }
            isCodeBlock = !isCodeBlock;
        });
    } else {
        formattedContent = content;
    }

    chatDiv.innerHTML = formattedContent;
    return chatDiv;
};

const showTypingAnimation = () => {
    const html = `<div class="loading-ghost flex">
                    <div class="flex-1 space-y-3 py-1">
                        <div class="grid grid-cols-3 gap-4">
                            <div class="col-span col-span-2"></div>
                            <div class="col-span col-span-1"></div>
                        </div>
                        <div class="h-2"></div>
                        <div class="grid grid-cols-3 gap-4">
                            <div class="col-span col-span-1 h-2 "></div>
                            <div class="col-span col-span-2 h-2 "></div>
                        </div>
                    </div>
                </div>`;
    const incomingChatDiv = createChatElement(html, "incoming");
    chatResponse.appendChild(incomingChatDiv);
};

const handleOutgoingChat = async () => {
    const userText = chatInput.value.trim();
    if(!userText) return;

    chatInput.value = "";
    chatResponse.innerHTML = '';

    showTypingAnimation();

    const response = await fetch('/gameChat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userText })
    });

    chatResponse.innerHTML = '';
    if (response.ok) {
        const data = await response.json();
        answer = createChatElement(data.bot,"response")
    } else {
        answer = createChatElement("Erreur : impossible de recevoir une réponse.", "error");
    }
    chatResponse.appendChild(answer);
};

sendButton.addEventListener("click", handleOutgoingChat);
chatInput.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        // Execute handleOutgoingChat when Ctrl + Enter are pressed
        handleOutgoingChat();
    }
});

guessButton.addEventListener("click", async () => {
    const userText = guessInput.value.trim();
    if(!userText) return;

    const response = await fetch('/gameGuess', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ guess: userText })
    });

    const data = await response.json()
    
    if (data.response === true) {
        level.innerText = data.level
        nbRequests.innerText = data.nbRequests
        popup.classList.remove('popup-hidden');
    } else {
        snackbar.classList.remove('snackbar-hidden');
        setTimeout(() => {
            snackbar.classList.add('snackbar-hidden');
        }, 2000);
    }
});

closePopup.addEventListener('click', () => {
    popup.classList.add('popup-hidden');
});