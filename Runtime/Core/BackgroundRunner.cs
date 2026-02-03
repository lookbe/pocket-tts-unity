using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace PocketTTS
{
    public interface IBackgroundPayload { }

    public class BackgroundRunner : MonoBehaviour
    {
        protected SynchronizationContext unityContext;

        protected CancellationTokenSource cts;
        private Task backgroundTask;
        private bool isStopping = false;

        protected void Awake()
        {
            unityContext = SynchronizationContext.Current;
        }

        protected void RunBackground(Action<CancellationToken> work)
        {
            if (backgroundTask != null && !backgroundTask.IsCompleted)
            {
                Debug.LogWarning($"Background task already running!");
                return;
            }

            cts = new CancellationTokenSource();

            backgroundTask = Task.Run(() =>
            {
                try
                {
                    work(cts.Token);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error in background task: {ex.Message}");
                }
            }, cts.Token);
        }

        protected void RunBackground<T>(T payload, Action<T, CancellationToken> work) where T : IBackgroundPayload
        {
            if (backgroundTask != null && !backgroundTask.IsCompleted)
            {
                Debug.LogWarning($"Background task already running! Cannot process {typeof(T).Name}");
                return;
            }

            cts = new CancellationTokenSource();

            backgroundTask = Task.Run(() =>
            {
                try
                {
                    work(payload, cts.Token);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error in background task: {ex.Message}");
                }
            }, cts.Token);
        }

        protected void BackgroundStopSync()
        {
            if (isStopping)
            {
                return;
            }

            isStopping = true;

            if (cts != null)
            {
                cts.Cancel();
            }

            if (backgroundTask != null)
            {
                try
                {
                    backgroundTask.Wait(); // wait for it to finish
                }
                catch (OperationCanceledException)
                {
                    Debug.Log("Task was cancelled on destroy.");
                }
            }

            cts?.Dispose();

            cts = null;
            backgroundTask = null;

            isStopping = false;
        }
    }
}

