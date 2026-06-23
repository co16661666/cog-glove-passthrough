using System;
using System.Reflection;
using UnityEngine;

/// <summary>
/// Attach to a Sphere GameObject. On Start it hides that sphere and uses it as the
/// scene root, spawning child spheres for every hand landmark and LineRenderers for
/// every bone connection. Reads HandTrackingFrame from a data-manager component via
/// reflection so you don't need a hard dependency on its concrete type.
///
/// Inspector setup:
///   1. Drag your global data-manager GameObject's component into "Data Manager".
///   2. (Optional) Assign a Joint Material and Bone Material, or leave blank for
///      auto-created Unlit materials.
///   3. Tweak Joint Radius, Bone Width, and per-hand colours as desired.
/// </summary>
[RequireComponent(typeof(MeshRenderer))]
public class HandSkeletonVisualizer : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Constants
    // ─────────────────────────────────────────────────────────────────────────

    private const int MAX_HANDS       = 2;
    private const int LANDMARK_COUNT  = 21;

    /// <summary>
    /// Landmark index pairs that form each bone segment.
    /// Indices follow the MediaPipe / HandLandmarks layout:
    ///   0  = Wrist
    ///   1–4  = Thumb  (CMC, MCP, IP, Tip)
    ///   5–8  = Index  (MCP, PIP, DIP, Tip)
    ///   9–12 = Middle (MCP, PIP, DIP, Tip)
    ///  13–16 = Ring   (MCP, PIP, DIP, Tip)
    ///  17–20 = Pinky  (MCP, PIP, DIP, Tip)
    /// </summary>
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

    [Header("Data Source")]
    [Tooltip("Drag the MonoBehaviour that exposes 'LatestHandFrame' here.")]
    [SerializeField] private MonoBehaviour dataManager;

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

    // [handSlot][landmarkIndex]
    private GameObject[][] _jointObjects;
    // [handSlot][boneIndex]
    private LineRenderer[] [] _boneLines;

    // Reflection handle — avoids a hard compile-time dependency on the manager type.
    private PropertyInfo _frameProperty;

    // ─────────────────────────────────────────────────────────────────────────
    // Unity lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        // The sphere this script lives on acts only as a scene-graph root;
        // hide its own renderer so it doesn't appear in the scene.
        GetComponent<MeshRenderer>().enabled = false;

        // Bind to data manager via reflection.
        if (dataManager != null)
        {
            _frameProperty = dataManager.GetType()
                                        .GetProperty("LatestHandFrame",
                                                     BindingFlags.Public | BindingFlags.Instance);
        }

        if (_frameProperty == null)
        {
            Debug.LogError(
                "[HandSkeletonVisualizer] Could not find public property 'LatestHandFrame' " +
                "on the assigned data manager. Check the reference in the Inspector.");
        }

        BuildPooledObjects();
    }

    private void Update()
    {
        if (_frameProperty == null) return;

        HandTrackingFrame frame;
        try
        {
            // Unbox the struct value returned by reflection.
            frame = (HandTrackingFrame)_frameProperty.GetValue(dataManager);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[HandSkeletonVisualizer] Could not read frame: {e.Message}");
            return;
        }

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

        // Pre-create fallback materials once (avoids per-object allocation).
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

                // Remove physics; this is a pure visualizer.
                Destroy(go.GetComponent<SphereCollider>());

                // Apply material + colour.
                var mr = go.GetComponent<MeshRenderer>();
                mr.sharedMaterial = jointMaterial != null
                    ? jointMaterial
                    : fallbackJoint;
                // Instance the material so each hand can have its own colour.
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

                // Instance material so colour is independent per hand.
                lr.material = boneMaterial != null
                    ? new Material(boneMaterial)
                    : fallbackBone;
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

    /// <summary>
    /// Flattens HandLandmarks into an ordered array matching BoneConnections indices.
    /// </summary>
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
        // "Sprites/Default" is available in both Built-in and URP pipelines and
        // renders without lighting, which is ideal for a debug overlay.
        // For HDRP, swap "Sprites/Default" for "HDRP/Unlit".
        var shader = Shader.Find("Sprites/Default") ?? Shader.Find("Unlit/Color");
        var mat    = new Material(shader) { color = color };
        return mat;
    }

    private void SetAllObjectsActive(bool active)
    {
        for (int h = 0; h < MAX_HANDS; h++)
        {
            foreach (var go in _jointObjects[h])
                go.SetActive(active);

            foreach (var lr in _boneLines[h])
                lr.gameObject.SetActive(active);
        }
    }
}