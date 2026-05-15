// TCP client code adapted from https://medium.com/@rabeeqiblawi/implementing-a-basic-tcp-server-in-unity-a-step-by-step-guide-449d8504d1c5

// System imports
using System;
// using System.Text;
// using System.Net;
// using Unity.Collections.LowLevel.Unsafe;

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
    private readonly Queue<CVPose> m_receivedDataQueue = new Queue<CVPose>(); // Buffer for incoming data

    // Parameters
    [SerializeField] private RawImage m_image;
    [SerializeField] private int m_targetWidth;
    [SerializeField] private int m_targetHeight;
    [SerializeField] private PassthroughCameraAccess m_cameraAccess;
    [SerializeField] private GameObject m_interactiveCube;
    [SerializeField] private Text m_debugText;
    [SerializeField] private Material m_secureMaterial;
    [SerializeField] private Material m_defaultMaterial;

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
                        m_receivedDataQueue.Enqueue(result);
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

        lock (m_receivedDataQueue)
        {
            // Dequeue oldest element in queue for processing
            if (m_receivedDataQueue.Count > 0)
            {
                dataToProcess = m_receivedDataQueue.Dequeue();
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
    }

    private void OnDestroy()
    {
        if (m_processBuffer != null)
            Destroy(m_processBuffer);
        if (m_cpuTexture != null)
            Destroy(m_cpuTexture);
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