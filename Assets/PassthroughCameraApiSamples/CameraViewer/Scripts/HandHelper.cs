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
    public static HandLandmarks Parse(float[] data, int startIndex, Pose cameraPose)
    {
        HandLandmarks hand = new HandLandmarks();
        
        // 1. Hand Type (0.0 = left, 1.0 = right)
        hand.isLeft = Mathf.Approximately(data[startIndex], 0.0f);
        int idx = startIndex + 1;

        // Helper to extract sequential Vector3 points
        Vector3 GetV3()
        {
            // Vector3 v = new Vector3(-data[idx] / 1000f, -data[idx + 2] / 1000f, data[idx + 1] / 1000); // Convert units and coordinates (m -> mm, -z)
            Vector3 v = new Vector3(-data[idx] / 1000f, -data[idx + 2] / 1000f, data[idx + 1] / 1000); // Convert units (m -> mm)
            idx += 3;
            return (cameraPose.rotation * v) + cameraPose.position;
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
}