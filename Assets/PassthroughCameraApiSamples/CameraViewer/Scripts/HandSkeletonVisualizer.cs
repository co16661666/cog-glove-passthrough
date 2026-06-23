using System;
using UnityEngine;

/// <summary>
/// Attach to a Sphere GameObject. On Start it hides that sphere and uses it as the
/// scene root, spawning child spheres for every hand landmark and LineRenderers for
/// every bone connection. 
/// 
/// Efficiently updates via event subscription rather than polling.
/// </summary>
[RequireComponent(typeof(MeshRenderer))]
public class HandSkeletonVisualizer : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Constants
    // ─────────────────────────────────────────────────────────────────────────

    private const int MAX_HANDS       = 2;
    private const int LANDMARK_COUNT  = 21;

    private static readonly int[][] BoneConnections =
    {
        // Thumb
        new[] { 0,  1 }, new[] { 1,  2 }, new[] { 2,  3 }, new[] { 3,  4 },
        // Index
        new[] { 0,  5 }, new[] { 5,  6 }, new[] { 6,  7 }, new[] { 7,  8 },
        // Middle
        new[] { 0,  9 }, new[] { 9, 10 }, new[] {10, 11 }, new[] {11, 12 },
        // Ring
        new[] { 0, 13 }, new[] {13, 14 }, new[] {14, 15 }, new[] {15, 16 },
        // Pinky
        new[] { 0, 17 }, new[] {17, 18 }, new[] {18, 19 }, new[] {19, 20 },
        // Palm knuckle bar
        new[] { 5,  9 }, new[] { 9, 13 }, new[] {13, 17 },
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Inspector fields
    // ─────────────────────────────────────────────────────────────────────────
    
    // NOTE: The dataManager field has been completely removed!

    [Header("Visuals")]
    [Tooltip("World-space radius of each joint sphere.")]
    [SerializeField] private float jointRadius = 0.012f;

    [Tooltip("Width of bone line renderers.")]
    [SerializeField] private float boneWidth = 0.006f;

    [Tooltip("Optional. If null an Unlit/Color material is created automatically.")]
    [SerializeField] private Material jointMaterial;

    [Tooltip("Optional. If null an Unlit/Color material is created automatically.")]
    [SerializeField] private Material boneMaterial;

    [SerializeField] private Color leftHandColor  = new Color(0.20f, 0.85f, 1.00f);
    [SerializeField] private Color rightHandColor = new Color(0.20f, 1.00f, 0.45f);

    // ─────────────────────────────────────────────────────────────────────────
    // Runtime state
    // ─────────────────────────────────────────────────────────────────────────

    private GameObject[][] _jointObjects;
    private LineRenderer[][] _boneLines;

    // ─────────────────────────────────────────────────────────────────────────
    // Unity lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        GetComponent<MeshRenderer>().enabled = false;
        BuildPooledObjects();
    }

    private void OnEnable()
    {
        // 1. Subscribe to the event when this GameObject is turned on.
        // Every time the TCP client fires this event, our UpdateSkeleton method will run.
        TcpDataClient.OnHandFrameReceived += UpdateSkeleton;
    }

    private void OnDisable()
    {
        // 2. CRITICAL: Always unsubscribe when disabled or destroyed to prevent memory leaks!
        TcpDataClient.OnHandFrameReceived -= UpdateSkeleton;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Event Handler (Replaces the old Update loop)
    // ─────────────────────────────────────────────────────────────────────────

    private void UpdateSkeleton(HandTrackingFrame frame)
    {
        // Hide everything, then re-enable only what's needed this frame.
        SetAllObjectsActive(false);

        if (frame.hands == null) return;

        int handsToRender = Mathf.Min(frame.handCount, MAX_HANDS);
        for (int h = 0; h < handsToRender; h++)
        {
            HandLandmarks hand       = frame.hands[h];
            Vector3[]     positions  = GetOrderedLandmarks(hand);

            // ── Joint spheres ─────────────────────────────────────────
            for (int j = 0; j < LANDMARK_COUNT; j++)
            {
                _jointObjects[h][j].transform.position = positions[j];
                _jointObjects[h][j].SetActive(true);
            }

            // ── Bone lines ────────────────────────────────────────────
            for (int b = 0; b < BoneConnections.Length; b++)
            {
                _boneLines[h][b].SetPosition(0, positions[BoneConnections[b][0]]);
                _boneLines[h][b].SetPosition(1, positions[BoneConnections[b][1]]);
                _boneLines[h][b].gameObject.SetActive(true);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Object pool construction
    // ─────────────────────────────────────────────────────────────────────────

    private void BuildPooledObjects()
    {
        _jointObjects = new GameObject[MAX_HANDS][];
        _boneLines    = new LineRenderer[MAX_HANDS][];

        Material fallbackJoint = BuildFallbackMaterial(Color.white);
        Material fallbackBone  = BuildFallbackMaterial(Color.white);

        for (int h = 0; h < MAX_HANDS; h++)
        {
            Color handColor = h == 0 ? leftHandColor : rightHandColor;

            // ── Joint spheres ─────────────────────────────────────────
            _jointObjects[h] = new GameObject[LANDMARK_COUNT];
            for (int j = 0; j < LANDMARK_COUNT; j++)
            {
                GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = $"Hand{h}_Joint{j:D2}";
                go.transform.SetParent(transform, worldPositionStays: false);
                go.transform.localScale = Vector3.one * (jointRadius * 2f);

                Destroy(go.GetComponent<SphereCollider>());

                var mr = go.GetComponent<MeshRenderer>();
                mr.sharedMaterial = jointMaterial != null ? jointMaterial : fallbackJoint;
                mr.material.color = handColor;

                go.SetActive(false);
                _jointObjects[h][j] = go;
            }

            // ── Bone lines ────────────────────────────────────────────
            _boneLines[h] = new LineRenderer[BoneConnections.Length];
            for (int b = 0; b < BoneConnections.Length; b++)
            {
                GameObject go = new GameObject($"Hand{h}_Bone{b:D2}");
                go.transform.SetParent(transform, worldPositionStays: false);

                var lr = go.AddComponent<LineRenderer>();
                lr.positionCount  = 2;
                lr.startWidth     = boneWidth;
                lr.endWidth       = boneWidth;
                lr.useWorldSpace  = true;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lr.receiveShadows = false;

                lr.material = boneMaterial != null ? new Material(boneMaterial) : fallbackBone;
                lr.material.color = handColor;
                lr.startColor = handColor;
                lr.endColor   = handColor;

                go.SetActive(false);
                _boneLines[h][b] = lr;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static Vector3[] GetOrderedLandmarks(HandLandmarks h)
    {
        return new[]
        {
            /* 0  */ h.wrist,
            /* 1  */ h.thumbCmc,  /* 2  */ h.thumbMcp,  /* 3  */ h.thumbIp,  /* 4  */ h.thumbTip,
            /* 5  */ h.indexMcp,  /* 6  */ h.indexPip,  /* 7  */ h.indexDip, /* 8  */ h.indexTip,
            /* 9  */ h.midMcp,    /* 10 */ h.midPip,    /* 11 */ h.midDip,   /* 12 */ h.midTip,
            /* 13 */ h.ringMcp,   /* 14 */ h.ringPip,   /* 15 */ h.ringDip,  /* 16 */ h.ringTip,
            /* 17 */ h.pinkyMcp,  /* 18 */ h.pinkyPip,  /* 19 */ h.pinkyDip, /* 20 */ h.pinkyTip,
        };
    }

    private static Material BuildFallbackMaterial(Color color)
    {
        var shader = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
        var mat    = new Material(shader) { color = color };
        return mat;
    }

    private void SetAllObjectsActive(bool active)
    {
        for (int h = 0; h < MAX_HANDS; h++)
        {
            foreach (var go in _jointObjects[h]) go.SetActive(active);
            foreach (var lr in _boneLines[h]) lr.gameObject.SetActive(active);
        }
    }
}