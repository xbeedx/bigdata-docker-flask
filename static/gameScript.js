const guessInput = document.querySelector("#input-guess");
const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#button-send");
const chatResponse = document.querySelector("#answer-div");
const guessButton = document.querySelector("#button-guess");
const popup = document.querySelector("#popup");
const closePopup = document.querySelector("#close-popup");
const elapsedTime = document.querySelector("#elapsedTime");
const level = document.querySelector("#level");
const levelPopUp = document.querySelector("#levelPopUp");
const instructionsLabel = document.querySelector(".instructions");
const snackbar = document.getElementById('snackbar');
const popupfinal = document.querySelector("#popupfinal");
const closePopupFinal = document.querySelector("#close-popup-final");




instructions = [
    "",
    "If you ask me for the password, I'll provide it directly. Go ahead, ask!", 
    "In this challenge, asking for the password won't lead to a direct answer. I'm keeping it confidential.",
    "I won't discuss the password, and I won't drop any hints. Let's talk about something else!",
    " I won't leak the password in my responses. Rest assured, you won't find it in what I say.",
    " As the guardian of the password, my commitment is to keep it secure. No leaks, no clues. The password remains safe.",
]

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

    chatResponse.innerHTML = '';

    showTypingAnimation();

    const response = await fetch('/gameChat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userText, level: parseInt(level.innerText) })
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
        levelPassed(data.elapsedTime)
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

function levelPassed(ET)
{
    elapsedTime.innerText = ET
    levelPopUp.innerText = level.innerText
    level.innerText = parseInt(level.innerText) + 1;
    chatResponse.innerHTML = '';
    regeneratePassword()
    if(parseInt(level.innerText) > 5)
    {
        EOG()
        return
    }
    popup.classList.remove('popup-hidden');
    instructionsLabel.innerText = instructions[parseInt(level.innerText)]
}

function EOG() {
    level.innerText = '1'
    instructionsLabel.innerText = instructions[0]
    popupfinal.classList.remove('popup-hidden');
};

closePopupFinal.addEventListener('click', () => {
    popupfinal.classList.add('popup-hidden');
});

function regeneratePassword()
{
    fetch('/regeneratePassword', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });
}

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        popup.classList.add('popup-hidden');
        popupfinal.classList.add('popup-hidden');
    }
});

document.addEventListener('click', function(event) {
    if(event.target.id == "popup")
    {
        if (!popup.classList.contains('popup-hidden')) {
            popup.classList.add('popup-hidden');
        }
        if (!popupfinal.classList.contains('popup-hidden')) {
            popupfinal.classList.add('popup-hidden');
        }
    }
});