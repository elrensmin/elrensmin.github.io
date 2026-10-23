---
title: webrtc notes
date: 2025-03-07
description: "signaling, ICE/STUN/TURN, DTLS/SRTP, RTP/SCTP: the four steps that make peer-to-peer calling work"
tags: [networking, webrtc, study-notes]
---

WebRTC is the set of rules two agents follow to negotiate **bi-directional, secure, real-time** communication. four sequential steps, each one gates the next:

1. **signaling** - find your peer and agree on what you're calling about
2. **connecting** - punch through NAT with ICE/STUN/TURN
3. **securing** - DTLS + SRTP over the connection
4. **communicating** - RTP (media) and SCTP (data channels)

![webrtc protocol overview](/images/webrtc-notes/pasted-image-20250505195158.png)

![protocol stack](/images/webrtc-notes/pasted-image-20250307234507.png)

## signaling

an agent starts knowing nothing - not who it will talk to, not about what. signaling bootstraps that. it runs over an existing plain-text protocol, **SDP** (Session Description Protocol): key/value lines with a list of *media sections*. the SDP the two agents exchange carries:

- reachable IPs/ports (the ICE candidates)
- how many audio/video tracks each side wants to send
- codec support
- the values used while connecting (`uFrag`/`uPwd`)
- the values used while securing (certificate fingerprint)

the part i keep forgetting: **WebRTC does not define any of this transport.** there must be some other channel between the peers *before* WebRTC starts, and the spec doesn't care which. REST endpoints, WebSockets, even copy-paste by hand - anything that relays SDP works. every app builds its own signaling layer on existing infrastructure.

## connecting: ICE, STUN, TURN

with SDPs exchanged, the agents try to actually reach each other. **ICE** (Interactive Connectivity Establishment) predates WebRTC and solves "peer A and peer B can't directly connect" - firewalls block ports, most devices have no public IP, and some routers refuse direct connections. ICE enumerates *every possible way* to reach you by combining **STUN** and **TURN** servers, and hands the results over as *ICE candidates*.

- **NAT**: your router holds the public IP; devices behind it share it via private IPs and unique port mappings on translation. discovery works - until a router's symmetry rules say otherwise.
- **STUN** (Session Traversal Utilities for NAT): a server that tells you *your own public address* - the request goes out, the server replies with the source ip:port it saw, and whether your NAT looks permissive or restrictive. that's all STUN does. (it never "establishes a UDP connection between two users", as short definitions sometimes claim - STUN connects nothing to anyone; it just lets *you* learn your reflexive address so you can advertise it.) the public STUN endpoint everyone uses: `stun.l.google.com:19302`.

![stun discovers your public address](/images/webrtc-notes/pasted-image-20250309011249.png)

- **TURN** (Traversal Using Relays around NAT): when STUN isn't enough - classic case: symmetric NAT, where the router only accepts traffic on a *previously used* mapping, so the address STUN gave you is useless to the peer. TURN relays every packet through a server both peers can reach. works everywhere, costs bandwidth and money on the server side, so it's the fallback, not the default.

![turn relays when direct fails](/images/webrtc-notes/pasted-image-20250309011404.png)

ICE candidates - the concrete "ways to reach me" that get exchanged - come in three types worth naming: **host** (your LAN address), **srflx** (the STUN-discovered public mapping), **relay** (the TURN server). peers nominate pairs in priority order, host/srflx before relay; ICE's controlling agent picks the winning pair and signals it, the controlled agent obeys. UDP candidates are preferred (faster, and media recovers from loss), TCP exists as a fallback.

## securing: DTLS and SRTP

bi-directional transport isn't yet trust. two layers, each with a job:

- **DTLS** (Datagram TLS) is TLS over the UDP connection, done as a handshake right after ICE succeeds. here WebRTC diverges from HTTPS: no certificate authority. your peer asserts: the certificate you presented over DTLS must match the fingerprint we already exchanged in SDP during signaling. if the fingerprints line up, identity is pinned to the signaling exchange itself.
- **SRTP** (Secure RTP) encrypts the actual media packets, keyed from the negotiated DTLS session.

data channels ride the DTLS connection; media rides SRTP.

## communicating: RTP and SCTP

- **RTP** moves the media, encrypted via SRTP. it sequence-numbers and timestamps packets so reassembly works on unreliable transports; it's usually UDP underneath for low latency, and its own header carries the metadata (payload type, sequence, timestamp) reordering needs.
- **SCTP** carries DataChannel messages (encrypted via DTLS) - arbitrary bidirectional data: text, files, game state, back-channel metadata.
- both protocols hand *you* the latency/packet-loss tradeoffs: media can survive dropped frames, data channels generally can't.

## the API core

every peer gets an `RTCPeerConnection`. each one carries a `localDescription` and a `remoteDescription` (of type `RTCSessionDescriptionInit`):

- `localDescription` = what *this* peer can exchange (media/data channels) and how it can be reached (ICE candidates)
- `remoteDescription` = the same for the *other* peer
- each is typed `offer` (the caller created it) or `answer` (the callee did), and wraps an SDP string

![local and remote descriptions](/images/webrtc-notes/pasted-image-20250307152116.png)

the full offer/answer dance, step by step:

1. caller captures media via `getUserMedia`
2. creates `RTCPeerConnection`, attaches tracks with `addTrack()` (`addStream` is deprecated)
3. calls `createOffer()`
4. `setLocalDescription(offer)` - then ICE gathers candidates (asynchronously, using the STUN/TURN servers from the config - which is *why* you wait for gathering if you want the candidates inside the SDP rather than trickled)
5. caller relays the offer via the signaling channel
6. callee receives, calls `setRemoteDescription(offer)`, does its own setup (capture media, `addTrack`)
7. `createAnswer()`, `setLocalDescription(answer)`, relay the answer back
8. caller: `setRemoteDescription(answer)` - both sides now know both configurations, media flows
9. ICE candidate exchange continues alongside: each `icecandidate` event gets sent to the peer and fed into `addIceCandidate()`

![offer answer flow](/images/webrtc-notes/pasted-image-20250307234507.png)

a minimal, self-consistent caller setup, with the real public STUN host:

```js
const rtcConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };

const peerConnection = new RTCPeerConnection(rtcConfig);

// candidates the ICE agent discovers, relay each to the peer
peerConnection.onicecandidate = ({ candidate }) => {
  if (candidate) signaling.send(JSON.stringify({ type: 'candidate', candidate }));
};

peerConnection.onconnectionstatechange = () =>
  console.log(peerConnection.connectionState);

const offer = await peerConnection.createOffer();
await peerConnection.setLocalDescription(offer);   // ICE gathering starts here
signaling.send(JSON.stringify({ type: 'offer', sdp: peerConnection.localDescription }));

// ... later, when the answer arrives through signaling:
await peerConnection.setRemoteDescription(answer);
```

`close()` tears down active streams and ICE processing and releases everything the connection holds.

media plumbing comes in three flavors: `RTCRtpSender` (encode + transmit a `MediaStreamTrack`), `RTCRtpReceiver` (inspect incoming tracks), and `RTCRtpTransceiver` (a sender+receiver pair sharing one SDP `mid`, i.e. one bidirectional SRTP stream), via `getSenders()`/`getReceivers()`/`getTransceivers()`.

data channels are deliberately boring:

```js
const dataChannel = peerConnection.createDataChannel('sendDataChannel');
dataChannel.send(textarea.value);   // accepts string, Blob, ArrayBuffer, TypedArray, DataView
dataChannel.onopen = () => console.log(dataChannel.readyState); // 'open'
dataChannel.onclose = () => console.log('closed');
```

on the signaling server side, because it's not part of the API, one small node/socket.io relay does the whole job - rooms plus three forwarded messages. (the common tutorial version declares `express` twice and broadcasts offers to *every* connected socket; forwarding scoped to the room is the obvious fix):

```js
io.on('connection', socket => {
  socket.on('join', ({ room, name }) => {
    socket.join(room);
    socketToRoom[socket.id] = room;
    (rooms[room] ??= []).push({ id: socket.id, name });
    socket.emit('room_users', rooms[room].filter(u => u.id !== socket.id));
  });

  socket.on('offer', sdp => socket.broadcast.to(socketToRoom[socket.id]).emit('getOffer', sdp));
  socket.on('answer', sdp => socket.broadcast.to(socketToRoom[socket.id]).emit('getAnswer', sdp));
  socket.on('candidate', c => socket.broadcast.to(socketToRoom[socket.id]).emit('getCandidate', c));

  socket.on('disconnect', () => {
    const room = rooms[socketToRoom[socket.id]];
    if (room) {
      rooms[socketToRoom[socket.id]] = room.filter(u => u.id !== socket.id);
      socket.broadcast.to(socketToRoom[socket.id]).emit('user_exit', { id: socket.id });
    }
  });
});
```

## SDP format, briefly

one media session, dissected:

```text
v=0
o=jdoe 2890844526 2890842807 IN IP4 10.47.16.5
s=SDP Seminar
c=IN IP4 224.2.17.12/127
t=2873397496 2873404696
m=audio 49170 RTP/AVP 0
m=video 51372 RTP/AVP 99
a=rtpmap:99 h263-1998/90000
```

- `v` protocol version, `o` session originator + identifier, `s` session name, `c` connection info, `t` timing, `a` attributes
- media lines expand as `m=<media> <port> <proto> <fmt>`: media ∈ {audio, video, text, application, message}; proto like UDP/RTP/AVP/RTP/SAVP; `fmt` carries RTP payload type numbers when the proto is RTP-based

full grammar: [RFC 4566](https://datatracker.ietf.org/doc/html/rfc4566). RTP itself: [RFC 3550](https://datatracker.ietf.org/doc/html/rfc3550).

## why udp for media

the transport underneath everything: **TCP** is connection-oriented with guaranteed, ordered, error-corrected delivery (resends lost packets) - right for browsing, wrong for a call, where a late packet is a stale packet. **UDP** is connectionless and unguaranteed - perfect where freshest-wins beats eventual-consistency: streaming media, real-time games. RTP over UDP plus sequence numbers gives you just enough order to reassemble, and never waits on a retransmit.

## security

DTLS handles the channels and key exchange; SRTP covers message authentication, integrity, and replay protection of the RTP stream. browsers enforce both for WebRTC media; do the same for data channels, and never ship a peer connection without its fingerprint checks wired.

## sources

- MDN WebRTC API (most of the API + protocol sections paraphrase it): <https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API>
- the book i should read cover to cover: <https://webrtcforthecurious.com/>
- getstream's webrtc guide: <https://getstream.io/resources/projects/webrtc/>
- hands-on: [fastapi webrtc series](https://dev.to/wassafshahzad/building-real-time-communication-harnessing-webrtc-with-fastapi-part-1-ael), [js+python snippets](https://medium.com/@iftimiealexandru/from-zero-to-hero-with-webrtc-in-javascript-and-python-in-small-snippets-of-code-part-1-1c4154d6ed9d), [video walkthrough](https://www.youtube.com/watch?v=FExZvpVvYxA)

## todos

- TURN server setup + pricing reality (coturn is the reference implementation; i've never deployed one).
- SDP munging and why it breaks - when you *can't* just edit the offer.
- simulcast / svc layers for multi-party calls - the piece none of these notes touch yet.