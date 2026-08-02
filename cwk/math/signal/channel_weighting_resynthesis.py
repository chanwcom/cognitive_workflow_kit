import torch
import librosa

def small_energy_masking(mel_spec):
    mel_spec = mel_spec.to(mel_spec.device)
    peak_energy = torch.quantile(mel_spec, 0.95, dim=2, keepdim=True)

    eta_a = -80
    eta_b = 0
    eta_th = -20

    e_th = peak_energy * (10 ** (eta_th / 10))

    mask = (mel_spec >= e_th).float()

    masked_mel_spec = mel_spec * mask

    scaling_factor = torch.sum(mel_spec) / (torch.sum(masked_mel_spec) + 1e-9)
    masked_mel_spec *= scaling_factor

    return masked_mel_spec

def generate_adversarial_speech(original_speech): #, speaker_encoder, num_iterations=50, epsilon=0.02, alpha=0.0004):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_fft, hop_length, win_length = 512, 200, 400
    min_samples = 32000

    if original_speech.size(-1) < min_samples:
        original_speech = F.pad(original_speech, (0, min_samples - original_speech.size(-1)), mode='constant')

    stft = torch.stft(
        original_speech,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(device),
        return_complex=True,
        pad_mode='reflect',
        center=True,
        normalized=True
    )

    spec_mag = torch.abs(stft)
    spec_power = spec_mag ** 2
    phase = torch.angle(stft)

    mel_basis = torch.FloatTensor(librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=40)).to(device)
    mel_spec = torch.matmul(mel_basis, spec_power)
    spec_mel_mask = small_energy_masking(mel_spec)
    log_mask = torch.log(spec_mel_mask + 1e-9)
    # Get original embedding
    log_mel = torch.log(mel_spec.clamp(min=1e-12))
    spec_2d = log_mel.squeeze(0)

    import pdb; pdb.set_trace()
    e = speaker_encoder(spec_2d)
    e = F.normalize(e, p=2, dim=1)

    spec_adv = log_mel.clone()

    e_ = e.clone()
    for i in range(num_iterations):
        with torch.autocast(device_type='cuda', enabled=False):
            spec_2d = spec_adv.squeeze(0)
            e_tilde = speaker_encoder(spec_2d)
            e_tilde = F.normalize(e_tilde, p=2, dim=1)

            grad, loss = compute_gradient(e_, e_tilde)
            print(grad)
            print(grad.shape)
            grad = F.interpolate(
                grad.unsqueeze(0).unsqueeze(0),  # Makes it 4D [1, 1, 40, ...]
                size=(log_mask.shape[1], log_mask.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # Remove both added dimensions to get back to original shape


            grad = grad * log_mask

            perturbation = alpha * grad.sign()

            with torch.no_grad():
                spec_adv = spec_adv + perturbation
                spec_adv = torch.max(spec_adv - epsilon, torch.min(spec_adv + epsilon, spec_adv))
                spec_adv = torch.clamp(spec_adv, min=1e-6)

            e_ = e_tilde.clone()

    cosine_similarity = F.cosine_similarity(e, e_tilde).mean().item()

    mel_spec_adv = torch.exp(spec_adv)

    # Step 2: mel → linear spectrogram
    mel_basis_pinv = torch.linalg.pinv(mel_basis)  # pseudo-inverse of mel basis
    linear_spec_power = torch.matmul(mel_basis_pinv.to(device), mel_spec_adv)

    # Step 3: sqrt to get magnitude
    linear_spec_mag = torch.sqrt(linear_spec_power)

    # Step 4: combine magnitude with original phase
    stft_complex = torch.polar(linear_spec_mag, phase)  # mag + phase → complex

    # Step 5: inverse STFT
    waveform = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(device),
        length=original_speech.size(-1),  # match original length
        center=True,
        normalized=True
    )


    return waveform.detach(), cosine_similarity
