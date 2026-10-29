'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const players = new Map();
let activePlayer = null;
let featured = null;
let demoData = [];
const status = $('#audio-status');

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function time(value) {
  const s = Math.max(0, Math.floor(Number.isFinite(value) ? value : 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
function playIcon(playing) {
  return playing ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3h1v10H5zm5 0h1v10h-1z"/></svg>' : '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 3 8 5-8 5z"/></svg>';
}
function waveform(peaks) {
  return peaks.map((value, i) => {
    const height = Math.max(2, value * 64);
    return `<rect x="${i * 4}" y="${(72 - height) / 2}" width="2.3" height="${height}" rx="1.15"/>`;
  }).join('');
}
function playerHTML(sample, role, id) {
  const audio = sample[role];
  const name = `${role === 'original' ? 'Original' : 'SteerSound output'}: ${sample.title}`;
  const bars = waveform(audio.peaks);
  return `<div class="audio-player" data-player-id="${id}" role="group" aria-label="${escapeHTML(name)}"><button type="button" class="play-button" aria-label="Play ${escapeHTML(name)}" aria-pressed="false">${playIcon(false)}</button><div class="wave-track"><svg viewBox="0 0 640 72" preserveAspectRatio="none" aria-hidden="true">${bars}</svg><svg class="wave-played" viewBox="0 0 640 72" preserveAspectRatio="none" aria-hidden="true">${bars}</svg><input type="range" min="0" max="${audio.duration}" step="0.01" value="0" aria-label="Seek ${escapeHTML(name)}" aria-valuetext="0 seconds"></div><span class="audio-time" aria-hidden="true">0:00 / ${time(audio.duration)}</span><audio preload="none" src="${audio.src}" aria-hidden="true"></audio></div>`;
}
function bindPlayer(root, name) {
  const id = root.dataset.playerId;
  const audio = $('audio', root);
  const button = $('.play-button', root);
  const range = $('input', root);
  const fill = $('.wave-played', root);
  const clock = $('.audio-time', root);
  let duration = Number(range.max);
  let pendingSeek = null;
  let frame = null;
  let disposed = false;
  function update() {
    if (disposed) return;
    const value = pendingSeek === null ? audio.currentTime : pendingSeek;
    range.value = String(value);
    range.setAttribute('aria-valuetext', `${value.toFixed(1)} of ${duration.toFixed(1)} seconds`);
    fill.style.clipPath = `inset(0 ${100 - Math.min(100, value / duration * 100)}% 0 0)`;
    clock.textContent = `${time(value)} / ${time(duration)}`;
  }
  function tick() { update(); if (!audio.paused) frame = requestAnimationFrame(tick); }
  function setPlaying(playing) {
    button.innerHTML = playIcon(playing);
    button.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${name}`);
    button.setAttribute('aria-pressed', String(playing));
  }
  const controller = {
    audio,
    seek(value) {
      const target = Math.max(0, Math.min(value, duration));
      if (audio.readyState >= 1) { audio.currentTime = target; pendingSeek = null; }
      else { pendingSeek = target; audio.load(); }
      update();
    },
    async play() {
      if (activePlayer && activePlayer !== controller) activePlayer.audio.pause();
      activePlayer = controller;
      try { await audio.play(); }
      catch (error) {
        // A rapid version/task switch can intentionally interrupt a pending play.
        if (disposed || error.name === 'AbortError') return;
        status.textContent = `Could not play ${name}. Please use the WAV download link.`;
        setPlaying(false);
      }
    },
    dispose() { disposed = true; audio.pause(); cancelAnimationFrame(frame); audio.removeAttribute('src'); audio.load(); players.delete(id); if (activePlayer === controller) activePlayer = null; },
    get currentTime() { return pendingSeek === null ? audio.currentTime : pendingSeek; },
    get paused() { return audio.paused; },
  };
  button.addEventListener('click', () => audio.paused ? controller.play() : audio.pause());
  range.addEventListener('input', () => controller.seek(Number(range.value)));
  audio.addEventListener('loadedmetadata', () => {
    if (Number.isFinite(audio.duration)) { duration = audio.duration; range.max = String(duration); }
    if (pendingSeek !== null) { audio.currentTime = Math.min(pendingSeek, duration); pendingSeek = null; }
    update();
  });
  audio.addEventListener('play', () => {
    if (activePlayer && activePlayer !== controller) activePlayer.audio.pause();
    activePlayer = controller; setPlaying(true); cancelAnimationFrame(frame); tick();
    status.textContent = `Playing ${name}`;
  });
  audio.addEventListener('pause', () => { setPlaying(false); cancelAnimationFrame(frame); update(); });
  audio.addEventListener('ended', () => { setPlaying(false); cancelAnimationFrame(frame); update(); });
  audio.addEventListener('timeupdate', update);
  audio.addEventListener('error', () => {
    if (disposed) return;
    button.disabled = true; button.setAttribute('aria-label', `Audio unavailable: ${name}`);
    clock.textContent = 'Unavailable'; root.classList.add('audio-error');
  });
  players.set(id, controller);
  return controller;
}

function renderFeatured(role = 'edited', position = 0, resume = false) {
  if (featured) featured.dispose();
  const sample = demoData.find(d => d.id === 'remove1');
  $('#featured-player').innerHTML = playerHTML(sample, role, 'featured');
  featured = bindPlayer($('#featured-player .audio-player'), `${role === 'original' ? 'Original' : 'SteerSound output'}: remove background music`);
  if (position) featured.seek(position);
  if (resume) featured.play();
  $$('#hero-player [data-version]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.version === role)));
}

const taskDescriptions = {
  remove:'Remove the requested source while retaining the surrounding sound.',
  replace:'Replace the requested sound source while preserving the rest of the scene.',
  pitch:'Change the pitch of a specified source by an instructed number of semitones.',
  volume:'Change the level of a specified source without turning the whole scene up or down.',
  add:'Introduce a new sound into an existing recording.',
  generate:'Create audio from a text instruction. No input recording is required.',
};
function renderDemos(task) {
  $$('[data-player-id]', $('#demo-panel')).forEach(el => players.get(el.dataset.playerId)?.dispose());
  const samples = demoData.filter(d => d.task === task);
  $('#demo-panel').innerHTML = samples.map((sample, i) => {
    const row = role => `<div class="audio-row ${role}"><div class="player-label"><strong>${role === 'original' ? 'ORIGINAL' : 'STEERSOUND'}</strong><a class="download-link" href="${sample[role].src}" download aria-label="Download ${role === 'original' ? 'original' : 'SteerSound output'}: ${escapeHTML(sample.title)}">WAV <span aria-hidden="true">↓</span></a></div>${playerHTML(sample, role, `${sample.id}-${role}`)}</div>`;
    return `<article class="demo-card ${sample.original ? '' : 'generate-card'}"><div class="demo-card-top"><span class="small-label">${escapeHTML(sample.title)}</span><span>EXAMPLE ${String(i + 1).padStart(2,'0')}</span></div><h3>“${escapeHTML(sample.instruction)}”</h3>${sample.original ? '' : '<div class="generation-input">Text instruction → SteerSound → generated audio</div>'}<div class="card-players">${sample.original ? row('original') : ''}${row('edited')}</div></article>`;
  }).join('');
  $$('[data-player-id]', $('#demo-panel')).forEach(el => {
    const [sampleId, role] = [el.dataset.playerId.split('-')[0], el.dataset.playerId.split('-')[1]];
    const sample = samples.find(d => d.id === sampleId);
    bindPlayer(el, `${role === 'original' ? 'Original' : 'SteerSound output'}: ${sample.title}`);
  });
  $('#demo-panel').setAttribute('aria-labelledby', `tab-${task}`);
  $('#task-description').textContent = taskDescriptions[task];
}

function bindTabs(selector, dataKey, onChange) {
  const tabs = $$(selector);
  function select(tab, focus = false) {
    tabs.forEach(b => { const chosen = b === tab; b.setAttribute('aria-selected', String(chosen)); b.tabIndex = chosen ? 0 : -1; });
    onChange(tab.dataset[dataKey]);
    if (focus) tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (i + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); select(tabs[next], true); }
    });
  });
}

initializeResults();

fetch('data/demos.json', {cache:'no-cache'}).then(response=>{
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then(data=>{
  demoData=data;
  $('#demo-count').textContent=String(data.length);
  $$('.task-tabs [data-task]').forEach(tab=>{
    $('small',tab).textContent=String(data.filter(sample=>sample.task===tab.dataset.task).length);
  });
  renderFeatured(); renderDemos('remove');
  bindTabs('.task-tabs [role=tab]','task',renderDemos);
  $$('#hero-player [data-version]').forEach(button=>button.addEventListener('click',()=>{
    if(button.getAttribute('aria-pressed')==='true') return;
    const position=featured.currentTime;
    const resume=!featured.paused;
    renderFeatured(button.dataset.version,position,resume);
  }));
}).catch(()=>{
  $('#featured-player').innerHTML='<p class="audio-error">Audio gallery could not load. <a href="assets/audio/remove1-original.wav">Original WAV</a> · <a href="assets/audio/remove1-edited.wav">SteerSound WAV</a></p>';
  $('#demo-panel').innerHTML='<p class="audio-error">Please refresh to load the examples, or <a href="http://powersound.natapp1.cc/">try the online demo</a>.</p>';
});

const dialog=$('#figure-dialog');
let figureTrigger=null;
$$('[data-figure]').forEach(button=>button.addEventListener('click',()=>{
  figureTrigger=button;
  $('#figure-dialog-title').textContent=button.dataset.title;
  $('#enlarged-figure').src=button.dataset.figure;
  $('#enlarged-figure').alt=$('img',button).alt;
  dialog.showModal();
  $('#close-figure').focus();
}));
$('#close-figure').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}});
dialog.addEventListener('close',()=>figureTrigger?.focus());
document.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&!dialog.open&&activePlayer)activePlayer.audio.pause();
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&activePlayer)activePlayer.audio.pause();});
if('IntersectionObserver' in window){
  const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){$$('.nav-links a').forEach(a=>a.classList.toggle('active',a.hash===`#${entry.target.id}`));}}},{rootMargin:'-20% 0px -65% 0px'});
  $$('section[id]').filter(s=>['demos','research','method','benchmark','results'].includes(s.id)).forEach(s=>observer.observe(s));
}
