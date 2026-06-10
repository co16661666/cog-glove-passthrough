// TCP client code adapted from https://medium.com/@rabeeqiblawi/implementing-a-basic-tcp-server-in-unity-a-step-by-step-guide-449d8504d1c5

// System imports
using System;
// using System.Text;
// using System.Net;
// using Unity.Collections.LowLevel.Unsafe;

// For TCP communication
using System.Net.Sockets;
using System.Threading;
using System.IO;

// UnityEngine imports
using UnityEngine;

// using PassthroughCameraSamples;
using System.Collections.Generic;
public class TcpDataClient : MonoBehaviour
{
    // TCP Parameters
    [Header("TCP Network Settings")]
    [SerializeField] private string m_host = "127.0.0.1";
    [SerializeField] private int m_port = 65432;
    private TcpClient m_tcpClient;
    private NetworkStream m_stream;
    private Thread m_receiveThread;
    private bool m_isNetworkRunning = false;
    private readonly Queue<HandTrackingFrame> m_receiveQueue = new Queue<HandTrackingFrame>();
    // Public access to data
    public HandTrackingFrame LatestHandFrame { get; private set; }
    public static event Action<HandTrackingFrame> OnHandFrameReceived;
    public bool Grasped = false;

    private void Start()
    {
        ConnectToServer();
    }

    private void Update()
    {
        lock (m_receiveQueue)
        {
            while (m_receiveQueue.Count > 0)
            {
                HandTrackingFrame frame = m_receiveQueue.Dequeue();

                // Update the public property
                LatestHandFrame = frame;

                // Fire the event for any listening scripts
                OnHandFrameReceived?.Invoke(frame);
            }
        }
    }

    private void OnDestroy()
    {
        m_isNetworkRunning = false;

        if (m_stream != null) m_stream.Close();
        if (m_tcpClient != null) m_tcpClient.Close();
        if (m_receiveThread != null && m_receiveThread.IsAlive)
            m_receiveThread.Join(500);
    }

    private void ConnectToServer()
    {
        try
        {
            m_tcpClient = new TcpClient(m_host, m_port);
            m_stream = m_tcpClient.GetStream();
            m_isNetworkRunning = true;

            m_receiveThread = new Thread(ReceiveDataLoop)
            {
                IsBackground = true
            };
            m_receiveThread.Start();
            Debug.Log($"[TCP] Connected to server at {m_host}:{m_port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[TCP] Connection failed: {e.Message}");
        }
    }

    public void SendFrameOverNetwork(ulong timestamp, float[] floatData)
    {
        if (m_stream == null || !m_stream.CanWrite) return;

        try
        {
            BinaryWriter writer = new BinaryWriter(m_stream);

            // 1. Write Header Info
            writer.Write(timestamp);          // Header: Timestamp (ulong)
            writer.Write(floatData.Length);    // Header: Size / Number of floats (int)

            // 2. Convert float array directly into a compressed bit package (raw byte block)
            byte[] bytePackage = new byte[floatData.Length * sizeof(float)];
            Buffer.BlockCopy(floatData, 0, bytePackage, 0, bytePackage.Length);

            // 3. Send package
            writer.Write(bytePackage);
            m_stream.Flush();
        }
        catch (Exception e)
        {
            Debug.LogError($"[TCP] Send error: {e.Message}");
        }
    }

    private void ReceiveDataLoop()
    {
        BinaryReader reader = new BinaryReader(m_stream);

        while (m_isNetworkRunning && m_stream != null)
        {
            try
            {
                // 1. Parse incoming Header
                ulong timestamp = reader.ReadUInt64();
                int floatCount = reader.ReadInt32();

                // 2. Read the bit-packed array based on the float count
                int byteLength = floatCount * sizeof(float);
                byte[] byteBuffer = reader.ReadBytes(byteLength);

                if (byteBuffer.Length == byteLength)
                {
                    // Decompress the package of bits back into an array of floats
                    float[] receivedFloats = new float[floatCount]; // Inefficient, may need to switch to declaration outside of loop
                    Buffer.BlockCopy(byteBuffer, 0, receivedFloats, 0, byteLength);

                    if (receivedFloats.Length == 1)
                    {
                        Grasped = receivedFloats[0] == 1.0f;
                        continue;
                    }

                    HandTrackingFrame frame = new HandTrackingFrame
                    {
                        timestamp = timestamp
                    };

                    if (receivedFloats.Length > 0)
                    {
                        frame.handCount = Mathf.RoundToInt(receivedFloats[0]);
                        frame.hands = new HandLandmarks[frame.handCount];

                        for (int i = 0; i < frame.handCount; i++)
                        {
                            // Each hand block is exactly 64 floats long, starting after the initial count float
                            int startIndex = 1 + i * 64;
                            frame.hands[i] = HandLandmarks.Parse(receivedFloats, startIndex);
                        }
                    }
                    else
                    {
                        frame.handCount = 0;
                        frame.hands = Array.Empty<HandLandmarks>();
                    }

                    // Enqueue safely for processing on the Main Thread
                    lock (m_receiveQueue)
                    {
                        m_receiveQueue.Enqueue(frame);
                    }
                }
            }
            catch (Exception e)
            {
                if (m_isNetworkRunning)
                    Debug.LogError($"[TCP] Read/Disconnect error: {e.Message}");
                break;
            }
        }
    }
}