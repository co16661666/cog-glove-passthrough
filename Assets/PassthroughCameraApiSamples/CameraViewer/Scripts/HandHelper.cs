using System;
using UnityEngine;

[Serializable]
public struct HandLandmarks
{
    public bool isLeft; // true = left, false = right
    public Vector3 wrist;
    
    // Thumb
    public Vector3 thumbCmc;
    public Vector3 thumbMcp;
    public Vector3 thumbIp;
    public Vector3 thumbTip;

    // Index
    public Vector3 indexMcp;
    public Vector3 indexPip;
    public Vector3 indexDip;
    public Vector3 indexTip;

    // Middle
    public Vector3 midMcp;
    public Vector3 midPip;
    public Vector3 midDip;
    public Vector3 midTip;

    // Ring
    public Vector3 ringMcp;
    public Vector3 ringPip;
    public Vector3 ringDip;
    public Vector3 ringTip;

    // Pinky
    public Vector3 pinkyMcp;
    public Vector3 pinkyPip;
    public Vector3 pinkyDip;
    public Vector3 pinkyTip;

    // Parse 64-float byte chunk
    public static HandLandmarks Parse(float[] data, int startIndex, Pose cameraPose, Vector3 offset)
    {
        HandLandmarks hand = new HandLandmarks();
        
        // 1. Hand Type (0.0 = left, 1.0 = right)
        hand.isLeft = Mathf.Approximately(data[startIndex], 0.0f);
        int idx = startIndex + 1;

        // Helper to extract sequential Vector3 points
        Vector3 GetV3()
        {
            // Vector3 v = new Vector3(-data[idx] / 1000f, -data[idx + 2] / 1000f, data[idx + 1] / 1000); // Convert units and coordinates (m -> mm, -z)
            Vector3 v = new Vector3(-data[idx] / 1000f, -data[idx + 2] / 1000f, data[idx + 1] / 1000); // Convert units (m -> mm), coordinate change (Leap -> Unity)
            idx += 3;
            return (cameraPose.rotation * v) + (cameraPose.position + offset);
        }

        hand.wrist    = GetV3();
        
        hand.thumbCmc = GetV3();
        hand.thumbMcp = GetV3();
        hand.thumbIp  = GetV3();
        hand.thumbTip = GetV3();

        hand.indexMcp = GetV3();
        hand.indexPip = GetV3();
        hand.indexDip = GetV3();
        hand.indexTip = GetV3();

        hand.midMcp   = GetV3();
        hand.midPip   = GetV3();
        hand.midDip   = GetV3();
        hand.midTip   = GetV3();

        hand.ringMcp  = GetV3();
        hand.ringPip  = GetV3();
        hand.ringDip  = GetV3();
        hand.ringTip  = GetV3();

        hand.pinkyMcp = GetV3();
        hand.pinkyPip = GetV3();
        hand.pinkyDip = GetV3();
        hand.pinkyTip = GetV3();

        return hand;
    }
}

public struct HandTrackingFrame
{
    public ulong timestamp;
    public int handCount;
    public HandLandmarks[] hands;

    public float[] ToFlattenedFloatArray()
    {
        // Calculate exact total size:
        // 1 float (for handCount) + (64 floats per hand * handCount)
        int actualHandCount = (hands != null) ? hands.Length : 0;
        int totalSize = 1 + (actualHandCount * 64);
        
        float[] result = new float[totalSize];

        // 1. Index 0: Number of hands
        result[0] = (float)actualHandCount;

        // 2. Serialize each hand block
        int index = 1;
        for (int i = 0; i < actualHandCount; i++)
        {
            HandLandmarks hand = hands[i];

            // 1 float: Hand Type (0.0 = left, 1.0 = right)
            result[index++] = hand.isLeft ? 0.0f : 1.0f;

            // 63 floats: 21 landmark Vector3 values (x, y, z) in strict sequential order
            void WriteV3(Vector3 v)
            {
                result[index++] = v.x;
                result[index++] = v.y;
                result[index++] = v.z;
            }

            WriteV3(hand.wrist);

            // Thumb
            WriteV3(hand.thumbCmc);
            WriteV3(hand.thumbMcp);
            WriteV3(hand.thumbIp);
            WriteV3(hand.thumbTip);

            // Index
            WriteV3(hand.indexMcp);
            WriteV3(hand.indexPip);
            WriteV3(hand.indexDip);
            WriteV3(hand.indexTip);

            // Middle
            WriteV3(hand.midMcp);
            WriteV3(hand.midPip);
            WriteV3(hand.midDip);
            WriteV3(hand.midTip);

            // Ring
            WriteV3(hand.ringMcp);
            WriteV3(hand.ringPip);
            WriteV3(hand.ringDip);
            WriteV3(hand.ringTip);

            // Pinky
            WriteV3(hand.pinkyMcp);
            WriteV3(hand.pinkyPip);
            WriteV3(hand.pinkyDip);
            WriteV3(hand.pinkyTip);
        }

        return result;
    }
}