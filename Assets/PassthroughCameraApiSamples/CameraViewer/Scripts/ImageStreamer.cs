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

using Newtonsoft.Json;

// UnityEngine imports
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Assertions;
// using UnityEngine.Rendering;

// For passthrough camera
using System.Runtime.InteropServices;
using System.Collections;

using Meta.XR;
// using Meta.XR.Samples;
// using Meta.XR.EnvironmentDepth;

// using PassthroughCameraSamples;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
public class ImageStreamer : MonoBehaviour
{
    private readonly Queue<CVPose> m_poseQueue = new Queue<CVPose>();

    // Parameters
    [SerializeField] private RawImage m_image;
    [SerializeField] private int m_targetWidth;
    [SerializeField] private int m_targetHeight;
    [SerializeField] private PassthroughCameraAccess m_cameraAccess;
    [SerializeField] private GameObject m_interactiveCube;
    [SerializeField] private Text m_debugText;
    [SerializeField] private Material m_secureMaterial;
    [SerializeField] private Material m_defaultMaterial;
    // TCP Parameters
    [Header("TCP Network Settings")]
    [SerializeField] private string m_host = "127.0.0.1";
    [SerializeField] private int m_port = 65432;
    private TcpClient m_tcpClient;
    private NetworkStream m_stream;
    private Thread m_receiveThread;
    private bool m_isNetworkRunning = false;
    private readonly Queue<HandTrackingFrame> m_receiveQueue = new Queue<HandTrackingFrame>();
    // Public access to hand data
    public HandTrackingFrame LatestHandFrame { get; private set; }
    public static event Action<HandTrackingFrame> OnHandFrameReceived;

    [Header("Tuning Sensitivity")]
    private float m_sensitivity = 0.00001f;
    // One euro filter parameters
    private float m_minCutoffPosition = 0.70f;
    private float m_betaPosition = 10.0f;

    private float m_minCutoffRotation = 0.16f;
    private float m_betaRotation = 0.25f;

    private OneEuroVector3 m_positionFilter;
    private OneEuroQuaternion m_rotationFilter;

    private bool m_handshakeCompleted = true;

    private Vector3 m_adjustmentOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private bool m_euroAdjustment = false;

    // Image sending
    private DateTime m_startTime;
    private ulong m_timestamp = 0; // Timestamp of frame being processed

    // Double-buffering for GPU/CPU sync
    private RenderTexture m_processBuffer;  // Buffer being processed by C++
    private Texture2D m_cpuTexture;

    private long m_lastCameraTimestamp = 0;

    [StructLayout(LayoutKind.Sequential)]
    private struct CVPose
    {
        public float tx, ty, tz;
        public float rx, ry, rz;

        public int grasped;
        public int poseSuccess;
        public ulong timestamp;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct CameraPose
    {
        public float tx, ty, tz, rw, rx, ry, rz;
    }

    [DllImport("cv_wrapper")]
    private static extern unsafe void ProcessImage(
        void* imgData, int width, int height, ulong timestamp,
        CameraPose cam_pose, out CVPose result
    );

    [DllImport("cv_wrapper")]
    private static extern void RuntimeSetup(
        float fx, float fy, float cx, float cy,
        int targetWidth, int targetHeight,
        int sensorWidth, int sensorHeight
    );

    private IEnumerator Start()
    {
        var supportedResolutions = PassthroughCameraAccess.GetSupportedResolutions(PassthroughCameraAccess.CameraPositionType.Left);
        Assert.IsNotNull(supportedResolutions, nameof(supportedResolutions));
        Debug.Log($"PassthroughCameraAccess.GetSupportedResolutions(): {string.Join(", ", supportedResolutions)}");

        while (!m_cameraAccess.IsPlaying)
        {
            yield return null;
        }
        // Set texture to the RawImage Ui element
        m_image.texture = m_cameraAccess.GetTexture();
        m_startTime = m_cameraAccess.Timestamp;

        // Initialize double-buffering textures
        if (m_targetWidth > 0 && m_targetHeight > 0)
        {
            m_processBuffer = new RenderTexture(m_targetWidth, m_targetHeight, 0, RenderTextureFormat.R8);
            m_cpuTexture = new Texture2D(m_targetWidth, m_targetHeight, TextureFormat.R8, false);
        }

        // OpenCV setup
        RuntimeSetup(m_cameraAccess.Intrinsics.FocalLength.x, m_cameraAccess.Intrinsics.FocalLength.y,
                    m_cameraAccess.Intrinsics.PrincipalPoint.x, m_cameraAccess.Intrinsics.PrincipalPoint.y,
                    m_targetWidth, m_targetHeight,
                    m_cameraAccess.Intrinsics.SensorResolution.x, m_cameraAccess.Intrinsics.SensorResolution.y);

        Debug.Log("Runtime setup completed with intrinsics: fx=" + m_cameraAccess.Intrinsics.FocalLength.x +
                ", fy=" + m_cameraAccess.Intrinsics.FocalLength.y +
                ", cx=" + m_cameraAccess.Intrinsics.PrincipalPoint.x +
                ", cy=" + m_cameraAccess.Intrinsics.PrincipalPoint.y);

        // Connect to TCP server
        ConnectToServer();

        // Setup One Euro Filter
        // positionFilter = new OneEuroVector3(minCutoffPosition, betaPosition);
        // rotationFilter = new OneEuroQuaternion(minCutoffRotation, betaRotation);

        // float fx = m_cameraAccess.Intrinsics.FocalLength.x;
        // float fy = m_cameraAccess.Intrinsics.FocalLength.y;
        // float cx = m_cameraAccess.Intrinsics.PrincipalPoint.x;
        // float cy = m_cameraAccess.Intrinsics.PrincipalPoint.y;
    }

    private void Update()
    {
        if (m_handshakeCompleted && m_cameraAccess.IsPlaying)
        {
            Texture rawTexture = m_cameraAccess.GetTexture();
            if (rawTexture == null || m_processBuffer == null) return;

            if (m_cameraAccess.Timestamp.Ticks != m_lastCameraTimestamp)
            {
                m_lastCameraTimestamp = m_cameraAccess.Timestamp.Ticks;

                m_timestamp = (ulong)(m_cameraAccess.Timestamp - m_startTime).Ticks * 100; // Convert to nanoseconds
                Pose capturePose = m_cameraAccess.GetCameraPose();

                // Blit new frame to capture buffer (write)
                Graphics.Blit(rawTexture, m_processBuffer);
                GL.Flush();  // Ensure GPU finishes blitting

                RenderTexture.active = m_processBuffer;
                m_cpuTexture.ReadPixels(new Rect(0, 0, m_targetWidth, m_targetHeight), 0, 0);
                RenderTexture.active = null;

                unsafe
                {
                    NativeArray<byte> pixelData = m_cpuTexture.GetRawTextureData<byte>();
                    void* imgDataPtr = NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(pixelData);

                    var cameraPoseStruct = new CameraPose
                    {
                        tx = capturePose.position.x,
                        ty = capturePose.position.y,
                        tz = capturePose.position.z,
                        rw = capturePose.rotation.w,
                        rx = capturePose.rotation.x,
                        ry = capturePose.rotation.y,
                        rz = capturePose.rotation.z
                    };

                    ProcessImage(imgDataPtr, m_targetWidth, m_targetHeight, m_timestamp, cameraPoseStruct, out CVPose result);

                    Debug.Log("C++ processing completed for timestamp: " + m_timestamp);
                    if (result.poseSuccess != 0)
                    {
                        m_poseQueue.Enqueue(result);
                    }
                    Debug.Log("Result - Position: (" + result.tx + ", " + result.ty + ", " + result.tz + ") | Rotation: (" + result.rx + ", " + result.ry + ", " + result.rz + ")");
                }
            }
        }

        HandleControllerTuning();
    }

    private void LateUpdate()
    {
        // Check for new data in the thread-safe queue every frame
        CVPose dataToProcess = default;
        bool hasData = false;

        lock (m_poseQueue)
        {
            // Dequeue oldest element in queue for processing
            if (m_poseQueue.Count > 0)
            {
                dataToProcess = m_poseQueue.Dequeue();
                hasData = true;
            }
        }

        if (hasData)
        {
            Debug.Log("Server message received: " + JsonConvert.SerializeObject(dataToProcess));

            // 1. Convert OpenCV (RHS) to Unity (LHS)
            Vector3 worldPos = new Vector3(dataToProcess.tx, -dataToProcess.ty, dataToProcess.tz);

            Vector3 rotAxis = new Vector3(dataToProcess.rx, dataToProcess.ry, dataToProcess.rz);
            float angle = rotAxis.magnitude;
            Vector3 axis = rotAxis.normalized;
            Quaternion worldRot = Quaternion.AngleAxis(-angle * Mathf.Rad2Deg, new Vector3(axis.x, -axis.y, axis.z));

            bool isSecure = dataToProcess.grasped != 0;

            m_interactiveCube.transform.position = worldPos;
            m_interactiveCube.transform.rotation = worldRot;

            m_interactiveCube.GetComponent<Renderer>().material = isSecure ? m_secureMaterial : m_defaultMaterial;
        }

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
        if (m_processBuffer != null)
            Destroy(m_processBuffer);
        if (m_cpuTexture != null)
            Destroy(m_cpuTexture);

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
                    float[] receivedFloats = new float[floatCount];
                    Buffer.BlockCopy(byteBuffer, 0, receivedFloats, 0, byteLength);

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

    private void HandleControllerTuning()
    {
        if (OVRInput.GetDown(OVRInput.Button.One))
        {
            m_euroAdjustment = !m_euroAdjustment;
        }

        // A button
        // 1. Read Right Thumbstick (Position Tuning)
        Vector2 rightStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch);
        if (rightStick.magnitude > 0.1f)
        {
            if (m_euroAdjustment)
            {
                m_minCutoffPosition += rightStick.y * m_sensitivity * Time.deltaTime;
                m_betaPosition += rightStick.x * m_sensitivity * Time.deltaTime;

                // Clamping to prevent negative values
                m_minCutoffPosition = Mathf.Max(0.01f, m_minCutoffPosition);
                m_betaPosition = Mathf.Max(0.0f, m_betaPosition);

                m_positionFilter.UpdateParams(m_minCutoffPosition, m_betaPosition);
                Debug.Log($"POS TUNING: MinCutoff: {m_minCutoffPosition:F3} | Beta: {m_betaPosition:F3}");
            }
            else
            {
                m_adjustmentOffset.y += rightStick.y * m_sensitivity * Time.deltaTime;
                m_adjustmentOffset.z += rightStick.x * m_sensitivity * Time.deltaTime;
            }
        }

        // 2. Read Left Thumbstick (Rotation Tuning)
        Vector2 leftStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch);
        if (leftStick.magnitude > 0.1f)
        {
            if (m_euroAdjustment)
            {
                m_minCutoffRotation += leftStick.y * m_sensitivity * Time.deltaTime;
                m_betaRotation += leftStick.x * m_sensitivity * Time.deltaTime;

                m_minCutoffRotation = Mathf.Max(0.01f, m_minCutoffRotation);
                m_betaRotation = Mathf.Max(0.0f, m_betaRotation);

                m_rotationFilter.UpdateParams(m_minCutoffRotation, m_betaRotation);
                Debug.Log($"ROT TUNING: MinCutoff: {m_minCutoffRotation:F3} | Beta: {m_betaRotation:F3}");
            }
            else
            {
                m_adjustmentOffset.x += leftStick.x * m_sensitivity * Time.deltaTime;
            }
        }

        m_debugText.text = m_euroAdjustment
            ? "POS mincutoff: " + m_minCutoffPosition.ToString("F2") + "  |  " + "POS beta: " + m_betaPosition.ToString("F2")
                            + "\n" +
                         "ROT mincutoff: " + m_minCutoffRotation.ToString("F2") + "  |  " + "ROT beta: " + m_betaRotation.ToString("F2")
            : "X offset: " + m_adjustmentOffset.x.ToString("F5") + "  |  " + "Y offset: " + m_adjustmentOffset.y.ToString("F5") + "  |  " + "Z offset: " + m_adjustmentOffset.z.ToString("F5");
    }
}