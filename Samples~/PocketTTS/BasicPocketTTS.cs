using PocketTTS;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class BasicPocketTTS : MonoBehaviour
{
    public PocketTTS.PocketTTS tts;

    public TMP_Text chatHistory;
    public TMP_InputField chatInputField;
    public Button sendButton;

    void Start()
    {
        tts.InitModel();
        sendButton.onClick.AddListener(OnSendButtonClicked);
    }

    private void OnEnable()
    {
        if (tts != null)
        {
            tts.OnStatusChanged += OnBotStatusChanged;

            OnBotStatusChanged(tts.status);
        }
    }

    private void OnDisable()
    {
        if (tts != null)
        {
            tts.OnStatusChanged -= OnBotStatusChanged;
        }
    }

    void OnBotStatusChanged(ModelStatus status)
    {
        switch (status)
        {
            case ModelStatus.Loading:
                {
                    sendButton.interactable = false;
                }
                break;
            case ModelStatus.Ready:
                {
                    sendButton.GetComponentInChildren<TMP_Text>().text = "SEND";
                    sendButton.interactable = true;
                    ClearInput();
                }
                break;
            case ModelStatus.Generate:
                {
                    sendButton.GetComponentInChildren<TMP_Text>().text = "STOP";
                }
                break;
            case ModelStatus.Error:
                {
                    sendButton.interactable = true;
                }
                break;
        }
    }

    protected virtual void ClearInput()
    {
        chatInputField.text = "";
    }

    public void OnSendButtonClicked()
    {
        if (tts)
        {
            if (tts.status == ModelStatus.Ready)
            {
                string message = chatInputField.text;
                if (!string.IsNullOrEmpty(message))
                {
                    chatHistory.text += "tts: " + message + "\n";
                    tts.Prompt(message);
                    ClearInput();
                }
            }
            else
            {
                sendButton.interactable = false;
                //tts.Stop();
            }
        }
    }
}
