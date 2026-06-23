using UnityEngine;

/// <summary>
/// Attach this to any GameObject with a Renderer (like a Cube or Sphere).
/// It listens to the TCP client and changes its color based on the 'Grasped' state.
/// </summary>
[RequireComponent(typeof(Renderer))]
public class ColorChange : MonoBehaviour
{
    [Header("Data Source")]
    [Tooltip("Target data client to pull the Grasped state from.")]
    [SerializeField] private TcpDataClient tcpClient;

    [Header("Color Settings")]
    [SerializeField] private Color normalColor = Color.white;
    [SerializeField] private Color graspedColor = Color.red;

    private Material _targetMaterial;
    private bool _lastGraspedState;

    private void Awake()
    {
        // Cache the material instance so we don't leak memory changing colors
        _targetMaterial = GetComponent<Renderer>().material;
        _targetMaterial.color = normalColor;
    }

    private void OnEnable()
    {
        // Listen for the network frames to know when data has updated
        TcpDataClient.OnHandFrameReceived += CheckGraspedState;
    }

    private void OnDisable()
    {
        // Unsubscribe to clean up references
        TcpDataClient.OnHandFrameReceived -= CheckGraspedState;
    }

    private void CheckGraspedState(HandTrackingFrame frame)
    {
        // Guard check: ensure we have a reference to the client instance
        if (tcpClient == null) return;

        // Pull the live state of the Grasped bool
        bool isGrasped = tcpClient.Grasped;

        // Only change color if the state actually flipped (prevents redundant updates)
        if (isGrasped != _lastGraspedState)
        {
            _lastGraspedState = isGrasped;
            _targetMaterial.color = isGrasped ? graspedColor : normalColor;
            
            Debug.Log($"[Visuals] Object color shifted. Grasped: {isGrasped}");
        }
    }
}