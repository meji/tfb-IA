async function generateImage() {
    const prompt = document.getElementById("prompt").value;
    const style = document.getElementById("style").value;
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");

    result.innerHTML = "";
    loading.style.display = "block";

    const response = await fetch("/generate_image/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt, style })
    });

    if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const img = document.createElement("img");
        img.src = url;
        result.appendChild(img);
    } else {
        const error = await response.json();
        result.innerHTML = `<p>Error: ${error.detail}</p>`;
    }

    loading.style.display = "none";
}
