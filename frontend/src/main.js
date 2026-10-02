import './style.css';

const apiStatus = document.querySelector('#api-status');
const setupNote = document.querySelector('#setup-note');
const callButton = document.querySelector('#call-button');
const callHint = document.querySelector('#call-hint');
const transcript = document.querySelector('#transcript');
const emptyMark = document.querySelector('#empty-mark');
const emptyMessage = document.querySelector('#empty-message');
const imageButton = document.querySelector('#image-button');
const screenButton = document.querySelector('#screen-button');
const imageInput = document.querySelector('#image-input');
const imageContext = document.querySelector('#image-context');
const sharedImage = document.querySelector('#shared-image');
const visualSummary = document.querySelector('#visual-summary');
const citationList = document.querySelector('#citation-list');
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
let room = null;
let connecting = false;
let liveKitEvents = null;
let sessionId = null;
let lastCustomerMessage = '';
let conversationHistory = [];
let imagePreviewUrl = null;

function setCallState(state, message) {
  callHint.textContent = message;
  const active = state === 'active';
  callButton.classList.toggle('active', active);
  callButton.disabled = state === 'connecting';
  callButton.setAttribute('aria-label', active ? 'End voice call' : 'Start voice call');
  callButton.title = active ? 'End voice call' : 'Start voice call';
  imageButton.disabled = !active;
  screenButton.disabled = !active;
  callButton.innerHTML = active
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 7 17 16M17 8l-9 9"/><path d="M7 4 4 7c.5 5.2 7.8 12.5 13 13l3-3-4-4-3 2c-1.5-.7-2.3-1.5-3-3l2-3-5-5Z"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"/><path d="M19 11a7 7 0 0 1-14 0M12 18v4m-4 0h8"/></svg>';
}

function addTranscriptLine(speaker, text) {
  if (!text.trim()) return;
  emptyMark.hidden = true;
  emptyMessage.hidden = true;
  const line = document.createElement('li');
  line.className = `transcript-line ${speaker === 'You' ? 'customer' : 'agent'}`;
  const label = document.createElement('span');
  label.className = 'transcript-speaker';
  label.textContent = speaker;
  const content = document.createElement('span');
  content.textContent = text;
  line.append(label, content);
  transcript.append(line);
  transcript.scrollTop = transcript.scrollHeight;
}

function listenToRoom(activeRoom) {
  activeRoom.on(liveKitEvents.RoomEvent.TrackSubscribed, (track) => {
    if (track.kind !== liveKitEvents.Track.Kind.Audio) return;
    const audioElement = track.attach();
    audioElement.className = 'remote-audio';
    audioElement.autoplay = true;
    document.body.append(audioElement);
  });

  activeRoom.on(liveKitEvents.RoomEvent.TrackUnsubscribed, (track) => {
    track.detach().forEach((element) => element.remove());
  });

  activeRoom.on(liveKitEvents.RoomEvent.TranscriptionReceived, (segments, participant) => {
    for (const segment of segments) {
      const isCustomer = participant?.identity === activeRoom.localParticipant.identity;
      if (!segment.final) {
        if (isCustomer) lastCustomerMessage = `${lastCustomerMessage} ${segment.text}`.trim();
        continue;
      }
      addTranscriptLine(isCustomer ? 'You' : 'EchoDesk', segment.text);
      conversationHistory.push({ role: isCustomer ? 'user' : 'assistant', content: segment.text });
      if (isCustomer) lastCustomerMessage = segment.text;
    }
  });

  activeRoom.on(liveKitEvents.RoomEvent.DataReceived, (payload, participant) => {
    if (participant?.identity !== activeRoom.localParticipant.identity) return;
    try {
      const text = new TextDecoder().decode(payload);
      const message = JSON.parse(text);
      if (message.type === 'user_turn' && typeof message.text === 'string') {
        lastCustomerMessage = message.text;
        conversationHistory.push({ role: 'user', content: message.text });
      }
    } catch {
      return;
    }
  });

  activeRoom.on(liveKitEvents.RoomEvent.ParticipantConnected, () => {
    setCallState('active', 'Connected. Speak naturally; you can interrupt any time.');
  });

  activeRoom.on(liveKitEvents.RoomEvent.Disconnected, () => {
    room = null;
    setCallState('ready', 'Call ended. Start another when you’re ready.');
  });
}

async function startCall() {
  connecting = true;
  setCallState('connecting', 'Connecting your microphone…');
  let newRoom;
  try {
    sessionId = crypto.randomUUID();
    lastCustomerMessage = '';
    conversationHistory = [];
    const tokenResponse = await fetch(`${apiBaseUrl}/voice/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const token = await tokenResponse.json();
    if (!tokenResponse.ok) throw new Error(token.detail || 'Could not create a voice session.');

    const liveKit = await import('livekit-client');
    liveKitEvents = {
      RoomEvent: liveKit.RoomEvent,
      Track: liveKit.Track,
      ParticipantKind: liveKit.ParticipantKind,
    };
    newRoom = new liveKit.Room({ adaptiveStream: true, dynacast: true });
    room = newRoom;
    listenToRoom(newRoom);
    await newRoom.connect(token.server_url, token.participant_token);
    await newRoom.localParticipant.setMicrophoneEnabled(true);
    setCallState('active', 'Connected. Speak naturally; you can interrupt any time.');
    setupNote.textContent = 'LiveKit voice session connected.';
  } catch (error) {
    await newRoom?.disconnect();
    room = null;
    setCallState('ready', error instanceof Error ? error.message : 'Could not start the call.');
  } finally {
    connecting = false;
  }
}

async function endCall() {
  const activeRoom = room;
  room = null;
  if (activeRoom) await activeRoom.disconnect();
  sessionId = null;
  document.querySelectorAll('.remote-audio').forEach((element) => element.remove());
  setCallState('ready', 'Call ended. Start another when you’re ready.');
}

imageButton.addEventListener('click', () => imageInput.click());

async function shareImage(file) {
  if (!file || !room || !sessionId) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    callHint.textContent = 'Choose a PNG, JPEG, or WebP image.';
    return false;
  }
  if (file.size > 4 * 1024 * 1024) {
    callHint.textContent = 'Choose an image smaller than 4 MB.';
    return false;
  }

  if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  imagePreviewUrl = URL.createObjectURL(file);
  sharedImage.src = imagePreviewUrl;
  sharedImage.alt = `Shared image: ${file.name}`;
  imageContext.hidden = false;
  visualSummary.textContent = 'Analyzing image…';
  citationList.replaceChildren();
  callHint.textContent = 'Analyzing the shared image and checking support guidance…';
  imageButton.disabled = true;
  screenButton.disabled = true;

  const form = new FormData();
  form.set('image', file);
  form.set('session_id', sessionId);
  form.set('message', lastCustomerMessage || 'Please inspect this image for my support issue.');
  const previousHistory = [...conversationHistory];
  const latestTurn = previousHistory.at(-1);
  if (latestTurn?.role === 'user' && latestTurn.content === lastCustomerMessage) previousHistory.pop();
  form.set('history', JSON.stringify(previousHistory.slice(-60)));

  try {
    const response = await fetch(`${apiBaseUrl}/vision/analyze`, { method: 'POST', body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Image analysis could not be completed.');

    visualSummary.textContent = result.visual_summary || 'No visual findings were returned.';
    for (const citation of result.citations || []) {
      const item = document.createElement('li');
      item.textContent = `${citation.title} · ${citation.source}`;
      citationList.append(item);
    }

    const agentParticipant = [...room.remoteParticipants.values()].find(
      (participant) => participant.kind === liveKitEvents.ParticipantKind.AGENT,
    );
    if (agentParticipant) {
      await room.localParticipant.performRpc({
        destinationIdentity: agentParticipant.identity,
        method: 'speak_reply',
        payload: JSON.stringify({ reply: result.reply }),
        responseTimeout: 10000,
      });
      callHint.textContent = result.status === 'pending_approval'
        ? 'Approval required. No account action was taken.'
        : 'Image analyzed. EchoDesk is responding on the call.';
    } else {
      addTranscriptLine('EchoDesk', result.reply);
      conversationHistory.push({ role: 'assistant', content: result.reply });
      callHint.textContent = 'Image analyzed. Agent audio is not connected.';
    }
  } catch (error) {
    visualSummary.textContent = error instanceof Error ? error.message : 'Image analysis failed.';
    callHint.textContent = 'Image analysis could not be completed.';
    return false;
  } finally {
    imageButton.disabled = !room;
    screenButton.disabled = !room;
  }
  return true;
}

imageInput.addEventListener('change', async () => {
  const file = imageInput.files?.[0];
  if (file) await shareImage(file);
  imageInput.value = '';
});

screenButton.addEventListener('click', async () => {
  if (!room) return;
  if (!navigator.mediaDevices?.getDisplayMedia) {
    callHint.textContent = 'Screen capture is not supported in this browser.';
    return;
  }

  let screenStream;
  try {
    screenButton.disabled = true;
    callHint.textContent = 'Choose a screen or window to share…';
    screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
    const video = document.createElement('video');
    video.srcObject = screenStream;
    await video.play();
    if (!video.videoWidth || !video.videoHeight) {
      await new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')?.drawImage(video, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Could not capture the selected screen.');
    await shareImage(new File([blob], 'screen-share.png', { type: 'image/png' }));
  } catch (error) {
    if (error instanceof Error && error.name !== 'NotAllowedError' && error.name !== 'AbortError') {
      callHint.textContent = error.message;
    } else {
      callHint.textContent = 'Screen share was cancelled.';
    }
  } finally {
    screenStream?.getTracks().forEach((track) => track.stop());
    screenButton.disabled = !room;
  }
});

callButton.addEventListener('click', async () => {
  if (connecting) return;
  if (room) await endCall();
  else await startCall();
});

try {
  const response = await fetch(`${apiBaseUrl}/health`);
  if (!response.ok) throw new Error(`API returned ${response.status}`);
  const health = await response.json();
  apiStatus.classList.add('connected');
  apiStatus.innerHTML = '<i></i> API connected';
  setupNote.textContent = `${health.service} is ready.`;
} catch {
  apiStatus.classList.add('disconnected');
  apiStatus.innerHTML = '<i></i> API offline';
  setupNote.textContent = 'Start the FastAPI server to connect this demo shell.';
}