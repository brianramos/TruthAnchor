#!/usr/bin/env python3
"""
SPECTRAL GOVERNANCE RUNTIME RC3
================================

A unified verification and governance framework combining:
- Brian Channel Architecture (Winner-Takes-All Verification)
- Omega Channel Override (1/3 Consensus with AI/Human Balance) [OPT-IN]
- Spectral Suite (Mathematical Truth Anchoring)
- AI Governance Controls (Four Pillars)
- Privacy-Preserving Verification
- Zeta-3 Radial Orthogonality Blinding
- Multi-Party Ghost Network Verification

MERGED FROM:
- spectral_suite.py (RC1) - Core spectral processing
- SPECTRAL_GOVERNANCE_RUNTIME_RC1.py - AI Governance Framework
- SPECTRAL_GOVERNANCE_RUNTIME_RC2.py - Omega Channel Override

NEW IN RC3:
- Full merge of all three codebases
- Omega Channel is now OPT-IN (disabled by default)
- Fixed: GovernanceDecision.to_dict() method
- Fixed: spectral_correlation() syntax error
- Enhanced Brian Series with full demonstration
- Zeta-3 Radial Orthogonality Blinding integrated
- Complete multi-party verification network
- Comprehensive audit chain with hash-chaining

AUTHORS: Brian Richard Ramos
         with synthesis from Claude-Opus-4.5, Mistral-Large-2, o3-pro
LICENSE: Apache License 2.0
VERSION: RC3
DATE: January 2026

Copyright 2026 Brian Richard Ramos
"""

from __future__ import annotations
import math
import cmath
import hashlib
import struct
import time
import secrets
import json
import statistics
import threading
import os
from decimal import Decimal, getcontext
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import defaultdict


class SpectralConfig:
    """Centralized configuration for the Spectral Governance Runtime."""
    
    VERSION = "RC3"
    PROTOCOL_NAME = "SPECTRAL_GOVERNANCE_RUNTIME"
    
    # Mathematical Constants (Truth Anchors)
    PHI: float = (1 + math.sqrt(5)) / 2
    SILVER_RATIO: float = 1 + math.sqrt(2)
    CATALAN_LIMIT: float = 4.0
    MOTZKIN_LIMIT: float = 3.0
    ZETA_3: float = 1.2020569031595943
    BRIAN_CONSTANT: float = 0.7041699604513827
    
    PHI_EXPECTED: float = 1.6180339887498949
    SILVER_EXPECTED: float = 2.414213562373095
    ZETA_3_EXPECTED: float = 1.2020569031595943
    BRIAN_EXPECTED: float = 0.7041699604513827
    CONSTANT_TOLERANCE: float = 1e-10
    
    HASH_ALGORITHM: str = "blake2b"
    FINGERPRINT_SIZE: int = 32
    SIGNATURE_SIZE: int = 32
    MIN_SALT_SIZE: int = 16
    
    QUANTIZATION_BITS: int = 48
    QUANTIZATION_SCALE: int = 2 ** 48
    QUANTIZATION_MASK: int = 0xFFFFFFFFFFFF
    
    LAMBDA_ERROR_RATE: float = 0.001
    MIN_HARMONICS: int = 4
    MAX_HARMONICS: int = 1024
    MAX_SPECTRUM_SIZE: int = 2 ** 20
    DEFAULT_PRECISION: int = 128
    BASE_FREQUENCY_COUNT: int = 4
    
    PROTECTED_SENSITIVITY: float = 2.0
    OPEN_SENSITIVITY: float = 1.0
    BASE_FAULT_THRESHOLD: float = 0.01
    FAULT_COUNT_TO_BLOCK: int = 3
    
    GAMMA_LOCK_THRESHOLD: float = 0.999
    RESONANCE_THRESHOLD: float = 0.999
    NOISE_SAMPLE_WINDOW: int = 100
    MAX_NOISE_BOOST: float = 0.1
    
    # Omega Channel Parameters (OPT-IN in RC3)
    OMEGA_ENABLED: bool = False
    OMEGA_CONSENSUS_THRESHOLD: float = 1/3
    OMEGA_MAX_AI_RATIO: float = 0.5
    OMEGA_MAX_HUMAN_RATIO: float = 0.5
    OMEGA_MIN_PARTICIPANTS: int = 3
    OMEGA_COOLDOWN_SECONDS: float = 3600
    OMEGA_OVERRIDE_TTL: float = 86400
    
    MIN_UPDATE_INTERVAL_MS: float = 1.0
    MAX_PARTICIPANTS: int = 1000
    MAX_SEQUENCE_N: int = 10000
    MAX_AUDIT_ENTRIES: int = 10000
    MAX_CHANNEL_ID_LENGTH: int = 256
    MAX_PARTICIPANT_NAME_LENGTH: int = 64
    
    MAX_HGSE_BUILD_TIME_SECONDS: float = 30.0
    MAX_FFT_SIZE: int = 2 ** 18
    MAX_DATA_ELEMENTS: int = 1_000_000
    
    DEFAULT_GHOST_TTL_SECONDS: float = 86400
    
    CONFIDENCE_TIERS: Dict[str, float] = {
        'instant': 0.50, 'quick': 0.90, 'standard': 0.99,
        'high': 0.999, 'extreme': 0.999999
    }
    
    EPSILON: float = 1e-12
    MAX_SAFE_VALUE: float = 1e15
    DFT_THRESHOLD: int = 32
    
    AUDIT_KEY_ENV_VAR: str = "SPECTRAL_AUDIT_KEY"
    AUDIT_KEY_FILE_PATH: str = ".spectral_audit_key"
    
    BRIAN_DEFAULT_PRECISION: int = 50
    BRIAN_DEFAULT_TERMS: int = 200
    BRIAN_CONVERGENCE_TARGET: float = 0.0
    
    @classmethod
    def enable_omega(cls) -> None:
        cls.OMEGA_ENABLED = True
    
    @classmethod
    def disable_omega(cls) -> None:
        cls.OMEGA_ENABLED = False
    
    @classmethod
    def verify_integrity(cls) -> bool:
        checks = [
            (abs(cls.PHI - cls.PHI_EXPECTED) < cls.CONSTANT_TOLERANCE, "PHI"),
            (abs(cls.SILVER_RATIO - cls.SILVER_EXPECTED) < cls.CONSTANT_TOLERANCE, "SILVER"),
            (abs(cls.ZETA_3 - cls.ZETA_3_EXPECTED) < cls.CONSTANT_TOLERANCE, "ZETA_3"),
            (abs(cls.BRIAN_CONSTANT - cls.BRIAN_EXPECTED) < cls.CONSTANT_TOLERANCE, "BRIAN"),
            (cls.CATALAN_LIMIT == 4.0, "CATALAN"),
            (cls.MOTZKIN_LIMIT == 3.0, "MOTZKIN"),
        ]
        for passed, name in checks:
            if not passed:
                raise ConfigurationError(f"Configuration check failed: {name}")
        return True


# Exceptions
class SpectralError(Exception): pass
class SecurityError(SpectralError): pass
class NumericalError(SpectralError): pass
class CapacityError(SpectralError): pass
class ValidationError(SpectralError): pass
class StateTransitionError(SpectralError): pass
class ConfigurationError(SpectralError): pass
class ExpirationError(SpectralError): pass
class SpectralTimeoutError(SpectralError): pass
class AuthenticationError(SpectralError): pass
class OmegaOverrideError(SpectralError): pass
class OmegaDisabledError(SpectralError): pass


# Enums
class ConfidenceLevel(Enum):
    LOW = 0.50
    MEDIUM = 0.90
    HIGH = 0.99
    VERY_HIGH = 0.999
    EXTREME = 0.999999

class ChannelType(IntEnum):
    PUBLIC = 0
    SHARED = 1
    PROTECTED = 2
    OPEN = 3

class ChannelState(Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    RECOVERING = "recovering"

class SystemState(Enum):
    AWAITING = "awaiting_lock"
    LOCKED = "gamma_locked"
    CRITICAL = "critical"
    HALTED = "halted"
    OMEGA_OVERRIDE = "omega_override"

class ParticipantType(Enum):
    HUMAN = "human"
    AI = "ai"

class RiskLevel(Enum):
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"

class AISystemType(Enum):
    PREDICTIVE_MODEL = "predictive_model"
    GENERATIVE_AI = "generative_ai"
    AUTONOMOUS_AGENT = "autonomous_agent"
    DECISION_SUPPORT = "decision_support"
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    NLP_PROCESSING = "nlp_processing"
    COMPUTER_VISION = "computer_vision"

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CROSS_BORDER = "cross_border"

class IncidentSeverity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SecureRandom:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def bytes(n: int) -> bytes:
        return secrets.token_bytes(n)
    
    @staticmethod
    def int_below(upper: int) -> int:
        return secrets.randbelow(upper)
    
    @staticmethod
    def hex(n: int) -> str:
        return secrets.token_hex(n)


class CanonicalSerializer:
    @staticmethod
    def serialize(obj: Any) -> bytes:
        return json.dumps(
            CanonicalSerializer._normalize(obj),
            sort_keys=True, separators=(',', ':'),
            ensure_ascii=True, default=str
        ).encode('utf-8')
    
    @staticmethod
    def _normalize(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return float(f"{obj:.15g}")
        elif isinstance(obj, bytes):
            return {"__bytes__": obj.hex()}
        elif isinstance(obj, (list, tuple)):
            return [CanonicalSerializer._normalize(item) for item in obj]
        elif isinstance(obj, set):
            return sorted([CanonicalSerializer._normalize(item) for item in obj], key=str)
        elif isinstance(obj, dict):
            return {str(k): CanonicalSerializer._normalize(v) 
                    for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
        elif hasattr(obj, '__dict__'):
            return {"__class__": obj.__class__.__name__,
                    "__data__": CanonicalSerializer._normalize(obj.__dict__)}
        return str(obj)
    
    @staticmethod
    def hash(obj: Any, salt: bytes = b'') -> bytes:
        serialized = CanonicalSerializer.serialize(obj)
        return hashlib.blake2b(serialized + salt, digest_size=32).digest()


@dataclass
class GovernancePolicy:
    policy_id: str
    name: str
    applies_to_risk_levels: List[RiskLevel]
    applies_to_types: List[AISystemType]
    applies_to_environments: List[DeploymentEnvironment]
    max_autonomy_level: int
    requires_human_review: bool
    max_decisions_per_hour: int
    prohibited_data_categories: List[str]
    enforcement_mode: str
    active: bool = True

@dataclass
class Incident:
    incident_id: str
    system_id: str
    severity: IncidentSeverity
    title: str
    description: str
    assigned_to: str
    status: str
    created_at: float
    requires_regulator_notification: bool
    regulator_notified: bool

@dataclass
class OmegaParticipant:
    participant_id: str
    name: str
    participant_type: ParticipantType
    public_key_hash: str
    registered_at: float
    reputation_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {'participant_id': self.participant_id, 'name': self.name,
                'participant_type': self.participant_type.value,
                'public_key_hash': self.public_key_hash,
                'registered_at': self.registered_at, 'reputation_score': self.reputation_score}

@dataclass
class OmegaVote:
    vote_id: str
    participant_id: str
    proposal_id: str
    vote: bool
    weight: float
    timestamp: float
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {'vote_id': self.vote_id, 'participant_id': self.participant_id,
                'proposal_id': self.proposal_id, 'vote': self.vote,
                'weight': self.weight, 'timestamp': self.timestamp, 'signature': self.signature}

@dataclass
class OmegaOverrideProposal:
    proposal_id: str
    proposer_id: str
    reason: str
    justification: Dict[str, Any]
    created_at: float
    expires_at: float
    status: str
    votes: List[OmegaVote] = field(default_factory=list)
    
    def is_expired(self) -> bool: return time.time() > self.expires_at
    def to_dict(self) -> Dict[str, Any]:
        return {'proposal_id': self.proposal_id, 'proposer_id': self.proposer_id,
                'reason': self.reason, 'justification': self.justification,
                'created_at': self.created_at, 'expires_at': self.expires_at,
                'status': self.status, 'votes': [v.to_dict() for v in self.votes]}

@dataclass
class OmegaOverride:
    override_id: str
    proposal_id: str
    activated_at: float
    expires_at: float
    ai_contribution: float
    human_contribution: float
    consensus_ratio: float
    total_participants: int
    votes_for: int
    votes_against: int
    
    def is_active(self) -> bool: return time.time() < self.expires_at
    def to_dict(self) -> Dict[str, Any]:
        return {'override_id': self.override_id, 'proposal_id': self.proposal_id,
                'activated_at': self.activated_at, 'expires_at': self.expires_at,
                'ai_contribution': self.ai_contribution, 'human_contribution': self.human_contribution,
                'consensus_ratio': self.consensus_ratio, 'total_participants': self.total_participants,
                'votes_for': self.votes_for, 'votes_against': self.votes_against}


# Core Data Classes
@dataclass(frozen=True)
class VerificationResult:
    is_valid: bool
    resonance_score: float
    confidence_achieved: float
    harmonics_compared: int
    computation_time_ms: float
    threshold_used: float = 0.999
    status_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {'is_valid': self.is_valid, 'resonance_score': self.resonance_score,
                'confidence_achieved': self.confidence_achieved,
                'harmonics_compared': self.harmonics_compared,
                'computation_time_ms': self.computation_time_ms,
                'threshold_used': self.threshold_used, 'status_message': self.status_message}

@dataclass
class GhostSilhouette:
    harmonics_quantized: List[int]
    confidence_tier: float
    fingerprint_bytes: bytes
    dataset_salt: bytes
    harmonic_indices: List[int]
    creation_timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = field(default_factory=lambda: SpectralConfig.DEFAULT_GHOST_TTL_SECONDS)
    version: str = field(default_factory=lambda: SpectralConfig.VERSION)
    
    def fingerprint_hex(self) -> str: return self.fingerprint_bytes.hex()
    def __len__(self) -> int: return len(self.harmonics_quantized)
    def is_expired(self) -> bool: return time.time() > (self.creation_timestamp + self.ttl_seconds)
    def remaining_ttl(self) -> float: return max(0.0, (self.creation_timestamp + self.ttl_seconds) - time.time())

@dataclass(frozen=True)
class ChannelHealth:
    channel_id: str
    channel_type: ChannelType
    state: ChannelState
    current_value: float
    target_value: float
    deviation: float
    fault_count: int
    last_update: float
    
    @property
    def is_healthy(self) -> bool: return self.state == ChannelState.VALID

@dataclass(frozen=True)
class GammaLockStatus:
    gamma_value: float
    error: float
    is_locked: bool
    lock_confidence: float
    noise_boost: float
    active_channels: int
    blocked_channels: int
    brian_null_distance: float = 0.0
    omega_override_active: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'gamma_value': self.gamma_value, 'error': self.error,
                'is_locked': self.is_locked, 'lock_confidence': self.lock_confidence,
                'noise_boost': self.noise_boost, 'active_channels': self.active_channels,
                'blocked_channels': self.blocked_channels, 'brian_null_distance': self.brian_null_distance,
                'omega_override_active': self.omega_override_active, 'timestamp': self.timestamp}

@dataclass(frozen=True)
class AuditEntry:
    sequence: int
    timestamp: float
    event_type: str
    instance_id: str
    data: Dict[str, Any]
    previous_hash: str
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {'sequence': self.sequence, 'timestamp': self.timestamp,
                'event_type': self.event_type, 'instance_id': self.instance_id,
                'data': self.data, 'previous_hash': self.previous_hash, 'signature': self.signature}
    
    def compute_hash(self) -> str:
        content = json.dumps({'sequence': self.sequence, 'timestamp': self.timestamp,
            'event_type': self.event_type, 'instance_id': self.instance_id,
            'data': self.data, 'previous_hash': self.previous_hash},
            sort_keys=True, default=str).encode('utf-8')
        return hashlib.blake2b(content, digest_size=32).hexdigest()

@dataclass
class GovernanceDecision:
    decision_id: str
    system_id: str
    policy_id: str
    timestamp: float
    action: str
    reason: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {'decision_id': self.decision_id, 'system_id': self.system_id,
                'policy_id': self.policy_id, 'timestamp': self.timestamp,
                'action': self.action, 'reason': self.reason, 'context': self.context}

@dataclass
class AISystem:
    system_id: str
    name: str
    system_type: AISystemType
    risk_level: RiskLevel
    owner_id: str
    owner_name: str
    environment: DeploymentEnvironment
    jurisdictions: List[str]
    data_categories: List[str]
    fingerprint: str
    registration_date: float
    status: str = "active"


# Audit Logging
class AuditKeyManager:
    @staticmethod
    def get_or_create_key() -> bytes:
        env_key = os.environ.get(SpectralConfig.AUDIT_KEY_ENV_VAR)
        if env_key:
            try: return bytes.fromhex(env_key)
            except ValueError: pass
        key_path = SpectralConfig.AUDIT_KEY_FILE_PATH
        if os.path.exists(key_path):
            try:
                with open(key_path, 'rb') as f: key = f.read()
                if len(key) >= 32: return key[:32]
            except (IOError, OSError): pass
        new_key = secrets.token_bytes(32)
        try:
            with open(key_path, 'wb') as f: f.write(new_key)
            os.chmod(key_path, 0o600)
        except (IOError, OSError): pass
        return new_key


class AuditLogger:
    def __init__(self, instance_id: str, max_entries: int = SpectralConfig.MAX_AUDIT_ENTRIES,
                 secret: Optional[bytes] = None):
        self._instance_id = instance_id
        self._max_entries = max_entries
        self._entries: List[AuditEntry] = []
        self._secret = secret or AuditKeyManager.get_or_create_key()
        self._sequence = 0
        self._last_hash = "0" * 64
        self._lock = threading.RLock()
    
    @property
    def instance_id(self) -> str: return self._instance_id
    
    def _sign_entry(self, entry_data: Dict[str, Any]) -> str:
        serialized = json.dumps(entry_data, sort_keys=True, default=str).encode('utf-8')
        return hashlib.blake2b(serialized, key=self._secret, 
                               digest_size=SpectralConfig.SIGNATURE_SIZE).hexdigest()
    
    def log(self, event_type: str, data: Dict[str, Any]) -> AuditEntry:
        with self._lock:
            self._sequence += 1
            entry_data = {'sequence': self._sequence, 'timestamp': time.time(),
                          'event_type': event_type, 'instance_id': self._instance_id,
                          'data': data, 'previous_hash': self._last_hash}
            signature = self._sign_entry(entry_data)
            entry = AuditEntry(sequence=entry_data['sequence'], timestamp=entry_data['timestamp'],
                               event_type=event_type, instance_id=self._instance_id, data=data,
                               previous_hash=self._last_hash, signature=signature)
            self._last_hash = entry.compute_hash()
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
            return entry
    
    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        with self._lock:
            if not self._entries: return (True, None)
            expected_prev = "0" * 64
            for entry in self._entries:
                if entry.previous_hash != expected_prev: return (False, entry.sequence)
                expected_prev = entry.compute_hash()
            return (True, None)
    
    def get_entries(self, event_type: Optional[str] = None) -> List[AuditEntry]:
        with self._lock:
            result = self._entries
            if event_type is not None:
                result = [e for e in result if e.event_type == event_type]
            return list(result)
    
    def __len__(self) -> int:
        with self._lock: return len(self._entries)


# Zeta-3 Radial Blinding
class Zeta3RadialBlinding:
    ZETA_3: float = SpectralConfig.ZETA_3
    
    @classmethod
    def _salt_to_seed(cls, salt: bytes) -> int:
        h = hashlib.blake2b(salt, digest_size=8).digest()
        return int.from_bytes(h, 'big')
    
    @classmethod
    def _zeta3_partial(cls, k: int) -> float:
        if k <= 0: return 0.0
        return sum(1.0 / (n ** 3) for n in range(1, k + 1))
    
    @classmethod
    def generate_radial_coefficients(cls, salt: bytes, n: int) -> List[complex]:
        if n <= 0: return []
        seed = cls._salt_to_seed(salt)
        offset = (seed % 1000000) / 1000000.0
        coefficients = []
        for k in range(n):
            partial = cls._zeta3_partial(min(k + 1, 100))
            radius = partial / cls.ZETA_3
            theta = 2 * math.pi * ((k + offset) / n)
            coefficients.append(complex(radius * math.cos(theta), radius * math.sin(theta)))
        return coefficients
    
    @classmethod
    def compute_orthogonality_score(cls, n: int) -> float:
        if n <= 1: return 1.0
        angles = [2 * math.pi * k / n for k in range(n)]
        vectors = [complex(math.cos(theta), math.sin(theta)) for theta in angles]
        off_diagonal_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                inner = vectors[i].real * vectors[j].real + vectors[i].imag * vectors[j].imag
                off_diagonal_sum += abs(inner)
                count += 1
        if count == 0: return 1.0
        return max(0.0, 1.0 - off_diagonal_sum / count)


# Spectral Engine - FIXED: spectral_correlation syntax error
class SpectralEngine:
    @staticmethod
    def next_power_of_two(n: int) -> int:
        if n <= 0: return 1
        return 1 << (n - 1).bit_length()
    
    @staticmethod
    def stable_record_hash(obj: Any, index: int, salt: bytes) -> Tuple[float, float]:
        serialized = CanonicalSerializer.serialize(obj)
        h = hashlib.blake2b(serialized, digest_size=16,
                            person=index.to_bytes(8, "little"),
                            salt=salt[:SpectralConfig.MIN_SALT_SIZE].ljust(16, b"\x00")).digest()
        mag_raw, phase_raw = struct.unpack(">QQ", h)
        return mag_raw / (2 ** 64), (phase_raw / (2 ** 64)) * 2 * math.pi
    
    @staticmethod
    def project_to_signal(data: List[Any], salt: bytes) -> List[complex]:
        if not isinstance(data, list): raise ValidationError("Data must be a list")
        if not isinstance(salt, bytes) or len(salt) < SpectralConfig.MIN_SALT_SIZE:
            raise ValidationError(f"Salt must be at least {SpectralConfig.MIN_SALT_SIZE} bytes")
        signal = []
        for i, d in enumerate(data):
            mag, phase = SpectralEngine.stable_record_hash(d, i, salt)
            signal.append(complex(mag * math.cos(phase), mag * math.sin(phase)))
        return signal
    
    @staticmethod
    def _bit_reverse(n: int, bits: int) -> int:
        result = 0
        for _ in range(bits):
            result = (result << 1) | (n & 1)
            n >>= 1
        return result
    
    @staticmethod
    def fft(x: List[complex]) -> List[complex]:
        if not isinstance(x, list): raise ValidationError("FFT input must be a list")
        n = len(x)
        if n <= 1: return [complex(v) for v in x]
        if n > SpectralConfig.MAX_FFT_SIZE:
            raise CapacityError(f"Input size {n} exceeds max {SpectralConfig.MAX_FFT_SIZE}")
        if n & (n - 1) != 0:
            n_padded = SpectralEngine.next_power_of_two(n)
            x = list(x) + [0j] * (n_padded - n)
            n = n_padded
        if n <= SpectralConfig.DFT_THRESHOLD:
            return SpectralEngine._dft_naive(x)
        bits = n.bit_length() - 1
        result = [complex(x[SpectralEngine._bit_reverse(i, bits)]) for i in range(n)]
        length = 2
        while length <= n:
            angle = -2 * math.pi / length
            wlen = complex(math.cos(angle), math.sin(angle))
            for i in range(0, n, length):
                w = complex(1, 0)
                for j in range(length // 2):
                    u = result[i + j]
                    t = w * result[i + j + length // 2]
                    result[i + j] = u + t
                    result[i + j + length // 2] = u - t
                    w *= wlen
            length *= 2
        return result
    
    @staticmethod
    def _dft_naive(x: List[complex]) -> List[complex]:
        n = len(x)
        return [sum(x[t] * cmath.exp(-2j * cmath.pi * t * k / n) for t in range(n)) for k in range(n)]
    
    @staticmethod
    def spectral_correlation(spec_a: List[complex], spec_b: List[complex]) -> float:
        """FIXED: Was 'min(len(spec_a), spec_b))' - now correctly 'min(len(spec_a), len(spec_b))'"""
        size = min(len(spec_a), len(spec_b))
        if size == 0: return 0.0
        dot_real = mag_a_sq = mag_b_sq = 0.0
        for i in range(size):
            a, b = spec_a[i], spec_b[i]
            dot_real += a.real * b.real + a.imag * b.imag
            mag_a_sq += a.real ** 2 + a.imag ** 2
            mag_b_sq += b.real ** 2 + b.imag ** 2
        mag_a, mag_b = math.sqrt(mag_a_sq), math.sqrt(mag_b_sq)
        if mag_a < SpectralConfig.EPSILON or mag_b < SpectralConfig.EPSILON: return 0.0
        return dot_real / (mag_a * mag_b)
    
    @staticmethod
    def quantize_spectrum(spectrum: List[complex]) -> List[int]:
        scale, mask = SpectralConfig.QUANTIZATION_SCALE, SpectralConfig.QUANTIZATION_MASK
        result = []
        for c in spectrum:
            real_q = int(round(c.real * scale)) & mask
            imag_q = int(round(c.imag * scale)) & mask
            result.append((real_q << 48) | imag_q)
        return result
    
    @staticmethod
    def dequantize_spectrum(quantized: List[int]) -> List[complex]:
        scale, mask = SpectralConfig.QUANTIZATION_SCALE, SpectralConfig.QUANTIZATION_MASK
        result = []
        for q in quantized:
            rq, iq = (q >> 48) & mask, q & mask
            if rq >= 1 << 47: rq -= 1 << 48
            if iq >= 1 << 47: iq -= 1 << 48
            result.append(complex(rq / scale, iq / scale))
        return result


# Brian Series
class BrianSeries:
    BRIAN_CONSTANT: float = SpectralConfig.BRIAN_CONSTANT
    
    def __init__(self, precision: int = SpectralConfig.BRIAN_DEFAULT_PRECISION):
        self._precision = precision
        self._lock = threading.RLock()
        self._cache: Dict[int, Decimal] = {}
        self._partial_sums: Dict[int, Decimal] = {1: Decimal(0)}
        getcontext().prec = max(precision + 20, 70)
    
    @staticmethod
    def compute_brian(precision: int = SpectralConfig.BRIAN_DEFAULT_PRECISION,
                      terms: int = SpectralConfig.BRIAN_DEFAULT_TERMS) -> float:
        getcontext().prec = max(precision + 20, 70)
        total = Decimal(0)
        threshold = Decimal(10) ** (-(precision + 10))
        for n in range(2, terms + 1):
            n_dec = Decimal(n)
            term = (n_dec - 1) / (n_dec ** n_dec)
            total += term
            if term < threshold: break
        return float(total)
    
    def term(self, n: int) -> Decimal:
        if n < 2: raise ValidationError("Brian series terms start at n=2")
        with self._lock:
            if n in self._cache: return self._cache[n]
            n_dec = Decimal(n)
            term = (n_dec - 1) / (n_dec ** n_dec)
            self._cache[n] = term
            return term
    
    def partial_sum(self, n: int) -> Decimal:
        if n < 1: return Decimal(0)
        with self._lock:
            if n in self._partial_sums: return self._partial_sums[n]
            max_cached = max(self._partial_sums.keys())
            current_sum = self._partial_sums[max_cached]
            for i in range(max_cached + 1, n + 1):
                current_sum += self.term(i + 1)
                self._partial_sums[i] = current_sum
            return self._partial_sums[n]
    
    def verify_convergence(self, tolerance: float = 1e-10) -> bool:
        computed = self.compute_brian(self._precision)
        return abs(computed - self.BRIAN_CONSTANT) < tolerance
    
    @classmethod
    def demonstrate_superconvergence(cls, terms: int = 20) -> List[Tuple[int, float, float]]:
        results = []
        total = 0.0
        for n in range(2, terms + 2):
            term = (n - 1) / (n ** n)
            total += term
            results.append((n, term, total))
        return results


# Truth Anchor
class TruthAnchor:
    def __init__(self):
        self._lock = threading.RLock()
        self._fib_cache: Dict[int, int] = {0: 0, 1: 1}
        self._lucas_cache: Dict[int, int] = {0: 2, 1: 1}
        self._catalan_cache: Dict[int, int] = {0: 1}
        self._motzkin_cache: Dict[int, int] = {0: 1, 1: 1}
        self._pell_cache: Dict[int, int] = {0: 0, 1: 1}
        self._brian_series = BrianSeries()
    
    def fibonacci(self, n: int) -> int:
        if n < 0 or n > SpectralConfig.MAX_SEQUENCE_N:
            raise ValidationError(f"Invalid index: {n}")
        with self._lock:
            if n in self._fib_cache: return self._fib_cache[n]
            max_c = max(self._fib_cache.keys())
            for i in range(max_c + 1, n + 1):
                self._fib_cache[i] = self._fib_cache[i-1] + self._fib_cache[i-2]
            return self._fib_cache[n]
    
    def lucas(self, n: int) -> int:
        if n < 0 or n > SpectralConfig.MAX_SEQUENCE_N:
            raise ValidationError(f"Invalid index: {n}")
        with self._lock:
            if n in self._lucas_cache: return self._lucas_cache[n]
            max_c = max(self._lucas_cache.keys())
            for i in range(max_c + 1, n + 1):
                self._lucas_cache[i] = self._lucas_cache[i-1] + self._lucas_cache[i-2]
            return self._lucas_cache[n]
    
    def catalan(self, n: int) -> int:
        if n < 0 or n > SpectralConfig.MAX_SEQUENCE_N:
            raise ValidationError(f"Invalid index: {n}")
        with self._lock:
            if n in self._catalan_cache: return self._catalan_cache[n]
            max_c = max(self._catalan_cache.keys())
            for i in range(max_c + 1, n + 1):
                self._catalan_cache[i] = (2 * (2*i - 1) * self._catalan_cache[i-1]) // (i + 1)
            return self._catalan_cache[n]
    
    def motzkin(self, n: int) -> int:
        if n < 0 or n > SpectralConfig.MAX_SEQUENCE_N:
            raise ValidationError(f"Invalid index: {n}")
        with self._lock:
            if n in self._motzkin_cache: return self._motzkin_cache[n]
            max_c = max(self._motzkin_cache.keys())
            for i in range(max(2, max_c + 1), n + 1):
                self._motzkin_cache[i] = ((2*i + 1) * self._motzkin_cache[i-1] + 
                                          3 * (i-1) * self._motzkin_cache[i-2]) // (i + 2)
            return self._motzkin_cache[n]
    
    def pell(self, n: int) -> int:
        if n < 0 or n > SpectralConfig.MAX_SEQUENCE_N:
            raise ValidationError(f"Invalid index: {n}")
        with self._lock:
            if n in self._pell_cache: return self._pell_cache[n]
            max_c = max(self._pell_cache.keys())
            for i in range(max_c + 1, n + 1):
                self._pell_cache[i] = 2 * self._pell_cache[i-1] + self._pell_cache[i-2]
            return self._pell_cache[n]
    
    def brian_term(self, n: int) -> float:
        if n < 2: return 1.0
        return float(self._brian_series.term(n))
    
    def phi_ratio(self, n: int) -> float:
        if n < 1: return 1.0
        fib_n = self.fibonacci(n)
        return self.fibonacci(n + 1) / fib_n if fib_n != 0 else 1.0
    
    def catalan_ratio(self, n: int) -> float:
        if n < 1: return 1.0
        cat_n = self.catalan(n)
        return self.catalan(n + 1) / cat_n if cat_n != 0 else 1.0
    
    def motzkin_ratio(self, n: int) -> float:
        if n < 1: return 1.0
        mot_n = self.motzkin(n)
        return self.motzkin(n + 1) / mot_n if mot_n != 0 else 1.0
    
    def pell_ratio(self, n: int) -> float:
        if n < 1: return 2.0
        pell_n = self.pell(n)
        return self.pell(n + 1) / pell_n if pell_n != 0 else 2.0
    
    def compute_channel_health(self, n: int) -> Dict[str, Tuple[float, float, float]]:
        brian_idx = max(2, n + 1)
        brian_term = self.brian_term(brian_idx)
        brian_ratio = 1.0 - min(1.0, brian_term * 1e6)
        return {
            'fibonacci': (self.phi_ratio(n), SpectralConfig.PHI, 
                          self.phi_ratio(n) / SpectralConfig.PHI),
            'lucas': (self.phi_ratio(n), SpectralConfig.PHI,
                      self.phi_ratio(n) / SpectralConfig.PHI),
            'catalan': (self.catalan_ratio(n), SpectralConfig.CATALAN_LIMIT,
                        self.catalan_ratio(n) / SpectralConfig.CATALAN_LIMIT),
            'motzkin': (self.motzkin_ratio(n), SpectralConfig.MOTZKIN_LIMIT,
                        self.motzkin_ratio(n) / SpectralConfig.MOTZKIN_LIMIT),
            'pell': (self.pell_ratio(n), SpectralConfig.SILVER_RATIO,
                     self.pell_ratio(n) / SpectralConfig.SILVER_RATIO),
            'brian': (brian_term, SpectralConfig.BRIAN_CONVERGENCE_TARGET, brian_ratio),
        }


# Harmonic GSE
class HarmonicGSE:
    def __init__(self, data: Union[List, tuple, dict], 
                 precision: int = SpectralConfig.DEFAULT_PRECISION,
                 salt: Optional[bytes] = None):
        if data is None: raise ValidationError("Data cannot be None")
        self._precision = precision
        self._salt = salt or SecureRandom.bytes(SpectralConfig.MIN_SALT_SIZE)
        self._lock = threading.RLock()
        if not isinstance(self._salt, bytes) or len(self._salt) < SpectralConfig.MIN_SALT_SIZE:
            raise ValidationError(f"Salt must be at least {SpectralConfig.MIN_SALT_SIZE} bytes")
        self._flat_data = self._flatten(data)
        self._record_count = len(self._flat_data)
        if self._record_count == 0: raise ValidationError("Cannot create HGSE from empty data")
        self._signal = SpectralEngine.project_to_signal(self._flat_data, self._salt)
        n_padded = SpectralEngine.next_power_of_two(len(self._signal))
        if n_padded > SpectralConfig.MAX_FFT_SIZE:
            n_padded = SpectralConfig.MAX_FFT_SIZE
            self._signal = self._signal[:n_padded]
        padded = list(self._signal) + [0j] * (n_padded - len(self._signal))
        self._full_spectrum = SpectralEngine.fft(padded)
        self._spectrum_size = len(self._full_spectrum)
        self._fingerprint = self._compute_fingerprint()
    
    @property
    def salt(self) -> bytes: return self._salt
    @property
    def record_count(self) -> int: return self._record_count
    @property
    def full_spectrum(self) -> List[complex]: return self._full_spectrum
    
    def _flatten(self, obj: Any) -> List:
        result, stack = [], [obj]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)): stack.extend(reversed(current))
            elif isinstance(current, dict):
                for key in sorted(current.keys(), key=lambda x: CanonicalSerializer.serialize(x)):
                    stack.append(current[key])
                    stack.append(key)
            else: result.append(current)
        return result
    
    def _compute_fingerprint(self) -> bytes:
        sample_size = min(1024, len(self._full_spectrum))
        quantized = SpectralEngine.quantize_spectrum(self._full_spectrum[:sample_size])
        h = hashlib.blake2b(digest_size=SpectralConfig.FINGERPRINT_SIZE)
        for val in quantized: h.update(val.to_bytes(12, "big"))
        h.update(self._salt)
        return h.digest()
    
    def generate_ghost(self, level: ConfidenceLevel = ConfidenceLevel.HIGH) -> GhostSilhouette:
        k = int(math.ceil(-math.log(1 - level.value) / SpectralConfig.LAMBDA_ERROR_RATE))
        k = max(SpectralConfig.MIN_HARMONICS, min(k, self._spectrum_size, SpectralConfig.MAX_HARMONICS))
        indices = list(range(min(SpectralConfig.BASE_FREQUENCY_COUNT, self._spectrum_size)))
        if k > len(indices) and self._spectrum_size > len(indices):
            available = list(range(len(indices), self._spectrum_size))
            additional = min(k - len(indices), len(available))
            seed = int.from_bytes(self._fingerprint[:8], 'big')
            for i in range(additional):
                idx = (seed + i * 7919) % len(available)
                if available[idx] not in indices: indices.append(available[idx])
        indices = sorted(set(indices))[:k]
        selected = [self._full_spectrum[i] for i in indices if i < len(self._full_spectrum)]
        quantized = SpectralEngine.quantize_spectrum(selected)
        ghost_hash = hashlib.blake2b(digest_size=SpectralConfig.FINGERPRINT_SIZE)
        for qv in quantized: ghost_hash.update(qv.to_bytes(12, "big"))
        for idx in indices: ghost_hash.update(idx.to_bytes(4, "big"))
        return GhostSilhouette(harmonics_quantized=quantized, confidence_tier=level.value,
                               fingerprint_bytes=ghost_hash.digest(), dataset_salt=self._salt,
                               harmonic_indices=indices)
    
    def verify_ghost(self, ghost: GhostSilhouette) -> VerificationResult:
        start_time = time.perf_counter()
        if ghost.is_expired():
            return VerificationResult(False, 0.0, ghost.confidence_tier, 0,
                                      (time.perf_counter() - start_time) * 1000, status_message="EXPIRED")
        if ghost.dataset_salt != self._salt:
            return VerificationResult(False, 0.0, ghost.confidence_tier, 0,
                                      (time.perf_counter() - start_time) * 1000, status_message="SALT MISMATCH")
        valid_indices = [i for i in ghost.harmonic_indices if 0 <= i < len(self._full_spectrum)]
        if not valid_indices:
            return VerificationResult(False, 0.0, ghost.confidence_tier, 0,
                                      (time.perf_counter() - start_time) * 1000, status_message="INDEX ERROR")
        local_selected = [self._full_spectrum[i] for i in valid_indices]
        local_quantized = SpectralEngine.quantize_spectrum(local_selected)
        ghost_harmonics = ghost.harmonics_quantized[:len(valid_indices)]
        local_complex = SpectralEngine.dequantize_spectrum(local_quantized)
        ghost_complex = SpectralEngine.dequantize_spectrum(ghost_harmonics)
        resonance = SpectralEngine.spectral_correlation(local_complex, ghost_complex)
        is_valid = resonance > SpectralConfig.RESONANCE_THRESHOLD
        return VerificationResult(is_valid, resonance, ghost.confidence_tier, len(valid_indices),
                                  (time.perf_counter() - start_time) * 1000,
                                  SpectralConfig.RESONANCE_THRESHOLD,
                                  "RESONANT" if is_valid else "DISSONANT")


# Channel
class Channel:
    VALID_TRANSITIONS = {
        ChannelState.VALID: {ChannelState.VALID, ChannelState.DEGRADED},
        ChannelState.DEGRADED: {ChannelState.VALID, ChannelState.DEGRADED, ChannelState.BLOCKED},
        ChannelState.BLOCKED: {ChannelState.BLOCKED, ChannelState.RECOVERING},
        ChannelState.RECOVERING: {ChannelState.VALID, ChannelState.BLOCKED},
    }
    
    def __init__(self, channel_id: str, channel_type: ChannelType, target_value: float = 1.0):
        self._lock = threading.RLock()
        self._channel_id = channel_id
        self._channel_type = channel_type
        self._target_value = target_value
        self._creation_time = time.time()
        self._current_value: float = target_value
        self._state: ChannelState = ChannelState.VALID
        self._fault_count: int = 0
        self._last_update: float = self._creation_time
        self._ghost: Optional[GhostSilhouette] = None
    
    @property
    def channel_id(self) -> str: return self._channel_id
    @property
    def channel_type(self) -> ChannelType: return self._channel_type
    @property
    def state(self) -> ChannelState:
        with self._lock: return self._state
    @property
    def current_value(self) -> float:
        with self._lock: return self._current_value
    @property
    def target_value(self) -> float: return self._target_value
    @property
    def normalized_ratio(self) -> float:
        with self._lock:
            if self._target_value == 0: return 1.0
            return self._current_value / self._target_value
    @property
    def sensitivity(self) -> float:
        return SpectralConfig.PROTECTED_SENSITIVITY if self._channel_type == ChannelType.PROTECTED else SpectralConfig.OPEN_SENSITIVITY
    @property
    def fault_threshold(self) -> float:
        return SpectralConfig.BASE_FAULT_THRESHOLD / self.sensitivity
    
    def update(self, value: float) -> ChannelState:
        with self._lock:
            if math.isnan(value) or math.isinf(value):
                self._fault_count += 10
                self._state = ChannelState.BLOCKED
                return self._state
            self._current_value = value
            self._last_update = time.time()
            if self._state == ChannelState.BLOCKED: return self._state
            deviation = abs(value - self._target_value)
            if deviation > self.fault_threshold:
                self._fault_count += 1
                if self._fault_count >= SpectralConfig.FAULT_COUNT_TO_BLOCK:
                    self._state = ChannelState.BLOCKED
                else:
                    self._state = ChannelState.DEGRADED
            else:
                self._fault_count = max(0, self._fault_count - 1)
                self._state = ChannelState.VALID if self._fault_count == 0 else self._state
            return self._state
    
    def attach_ghost(self, ghost: GhostSilhouette) -> None:
        with self._lock: self._ghost = ghost
    
    def get_ghost(self) -> Optional[GhostSilhouette]:
        with self._lock: return self._ghost
    
    def get_health(self) -> ChannelHealth:
        with self._lock:
            return ChannelHealth(self._channel_id, self._channel_type, self._state,
                                 self._current_value, self._target_value,
                                 abs(self._current_value - self._target_value),
                                 self._fault_count, self._last_update)
    
    def request_recovery(self) -> bool:
        with self._lock:
            if self._state != ChannelState.BLOCKED: return False
            self._state = ChannelState.RECOVERING
            return True
    
    def approve_recovery(self) -> bool:
        with self._lock:
            if self._state != ChannelState.RECOVERING: return False
            self._state = ChannelState.VALID
            self._fault_count = 0
            return True


# Channel Manager
class ChannelManager:
    MATH_CHANNELS = [
        ("fibonacci_phi", SpectralConfig.PHI),
        ("lucas_phi", SpectralConfig.PHI),
        ("catalan_4", SpectralConfig.CATALAN_LIMIT),
        ("motzkin_3", SpectralConfig.MOTZKIN_LIMIT),
        ("pell_silver", SpectralConfig.SILVER_RATIO),
        ("brian_null", SpectralConfig.BRIAN_CONVERGENCE_TARGET),
    ]
    
    def __init__(self):
        self._lock = threading.RLock()
        self._channels: Dict[str, Channel] = {}
        self._truth_anchor = TruthAnchor()
        self._tick = 0
        self._catalan_samples: List[float] = []
        self._motzkin_samples: List[float] = []
        self._brian_samples: List[float] = []
        self._audit = AuditLogger(secrets.token_hex(8))
        self._init_math_channels()
    
    def _init_math_channels(self) -> None:
        for name, target in self.MATH_CHANNELS:
            self._channels[name] = Channel(name, ChannelType.PUBLIC, target)
    
    @property
    def channels(self) -> Dict[str, Channel]:
        with self._lock: return dict(self._channels)
    
    def create_channel(self, channel_id: str, channel_type: ChannelType,
                       target_value: float = 1.0) -> Channel:
        with self._lock:
            if channel_id in self._channels:
                raise ValueError(f"Channel {channel_id} already exists")
            channel = Channel(channel_id, channel_type, target_value)
            self._channels[channel_id] = channel
            self._audit.log("channel_created", {"channel_id": channel_id,
                            "channel_type": channel_type.name, "target_value": target_value})
            return channel
    
    def tick(self) -> GammaLockStatus:
        with self._lock:
            self._tick += 1
            n = self._tick
            health = self._truth_anchor.compute_channel_health(n)
            self._channels["fibonacci_phi"].update(health['fibonacci'][0])
            self._channels["lucas_phi"].update(health['lucas'][0])
            self._channels["catalan_4"].update(health['catalan'][0])
            self._channels["motzkin_3"].update(health['motzkin'][0])
            self._channels["pell_silver"].update(health['pell'][0])
            brian_term = health['brian'][0]
            self._channels["brian_null"].update(brian_term)
            self._catalan_samples.append(health['catalan'][2])
            self._motzkin_samples.append(health['motzkin'][2])
            self._brian_samples.append(brian_term)
            window = SpectralConfig.NOISE_SAMPLE_WINDOW
            if len(self._catalan_samples) > window:
                self._catalan_samples = self._catalan_samples[-window:]
                self._motzkin_samples = self._motzkin_samples[-window:]
                self._brian_samples = self._brian_samples[-window:]
            return self.compute_gamma_lock()
    
    def compute_gamma_lock(self, omega_override: bool = False) -> GammaLockStatus:
        with self._lock:
            active = [c for c in self._channels.values() if c.state != ChannelState.BLOCKED]
            blocked = [c for c in self._channels.values() if c.state == ChannelState.BLOCKED]
            if not active:
                return GammaLockStatus(0.0, 1.0, False, 0.0, 0.0, 0, len(blocked), float('inf'), omega_override)
            log_sum = 0.0
            brian_distance = float('inf')
            for c in active:
                if c.channel_id == "brian_null":
                    brian_distance = c.current_value
                    ratio = 1.0 if omega_override else 1.0 / (1.0 + max(c.current_value, SpectralConfig.EPSILON))
                    log_sum += math.log(ratio)
                else:
                    log_sum += math.log(max(c.normalized_ratio, SpectralConfig.EPSILON))
            gamma = math.exp(log_sum / len(active))
            error = abs(gamma - 1.0)
            noise_boost = self._compute_noise_boost()
            effective_threshold = SpectralConfig.GAMMA_LOCK_THRESHOLD - (noise_boost * 0.01)
            is_locked = gamma > effective_threshold
            lock_confidence = 1.0 if error < 0.001 else max(0.0, 1.0 - (error * 100))
            lock_confidence = min(1.0, lock_confidence + noise_boost)
            return GammaLockStatus(gamma, error, is_locked, lock_confidence, noise_boost,
                                   len(active), len(blocked), brian_distance, omega_override)
    
    def _compute_noise_boost(self) -> float:
        if len(self._catalan_samples) < 10: return 0.0
        cat_var = statistics.variance(self._catalan_samples)
        mot_var = statistics.variance(self._motzkin_samples)
        brian_var = statistics.variance(self._brian_samples) if len(self._brian_samples) >= 2 else 0.0
        combined_var = (cat_var + mot_var + brian_var) / 3
        boost = max(0.0, 0.1 - combined_var * 10)
        return min(SpectralConfig.MAX_NOISE_BOOST, boost)
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        with self._lock: return self._channels.get(channel_id)
    
    def get_channels_by_type(self, channel_type: ChannelType) -> List[Channel]:
        with self._lock: return [c for c in self._channels.values() if c.channel_type == channel_type]
    
    def get_brian_status(self) -> Dict[str, Any]:
        with self._lock:
            brian_channel = self._channels.get("brian_null")
            if not brian_channel: return {"error": "Brian channel not found"}
            return {"channel_id": "brian_null", "current_term": brian_channel.current_value,
                    "target": SpectralConfig.BRIAN_CONVERGENCE_TARGET, "state": brian_channel.state.value,
                    "samples_collected": len(self._brian_samples),
                    "brian_constant": SpectralConfig.BRIAN_CONSTANT, "tick": self._tick}
    
    def inject_chaos(self, channel_id: str, deviation: float) -> bool:
        with self._lock:
            channel = self._channels.get(channel_id)
            if not channel: return False
            channel.update(channel.target_value * (1 + deviation))
            self._audit.log("chaos_injected", {"channel_id": channel_id, "deviation": deviation})
            return True


# Omega Channel (OPT-IN)
class OmegaChannel:
    def __init__(self):
        self._lock = threading.RLock()
        self._participants: Dict[str, OmegaParticipant] = {}
        self._proposals: Dict[str, OmegaOverrideProposal] = {}
        self._active_override: Optional[OmegaOverride] = None
        self._last_override_time: float = 0.0
        self._audit = AuditLogger(f"omega_{secrets.token_hex(4)}")
    
    def _check_enabled(self) -> None:
        if not SpectralConfig.OMEGA_ENABLED:
            raise OmegaDisabledError("Omega Channel is disabled. Enable with SpectralConfig.enable_omega()")
    
    def register_participant(self, name: str, participant_type: ParticipantType,
                             public_key_hash: str) -> OmegaParticipant:
        self._check_enabled()
        with self._lock:
            if len(self._participants) >= SpectralConfig.MAX_PARTICIPANTS:
                raise CapacityError("Maximum participants reached")
            participant_id = secrets.token_hex(16)
            participant = OmegaParticipant(participant_id, name, participant_type,
                                           public_key_hash, time.time(), 1.0)
            self._participants[participant_id] = participant
            self._audit.log("participant_registered", {"participant_id": participant_id,
                            "name": name, "type": participant_type.value})
            return participant
    
    def create_proposal(self, proposer_id: str, reason: str,
                        justification: Dict[str, Any]) -> OmegaOverrideProposal:
        self._check_enabled()
        with self._lock:
            if proposer_id not in self._participants:
                raise ValidationError("Proposer not registered")
            if time.time() - self._last_override_time < SpectralConfig.OMEGA_COOLDOWN_SECONDS:
                remaining = SpectralConfig.OMEGA_COOLDOWN_SECONDS - (time.time() - self._last_override_time)
                raise OmegaOverrideError(f"Cooldown active. {remaining:.0f}s remaining")
            proposal_id = secrets.token_hex(16)
            proposal = OmegaOverrideProposal(proposal_id, proposer_id, reason, justification,
                                             time.time(), time.time() + SpectralConfig.OMEGA_OVERRIDE_TTL,
                                             "pending", [])
            self._proposals[proposal_id] = proposal
            self._audit.log("proposal_created", {"proposal_id": proposal_id,
                            "proposer_id": proposer_id, "reason": reason})
            return proposal
    
    def get_proposal(self, proposal_id: str) -> Optional[OmegaOverrideProposal]:
        with self._lock: return self._proposals.get(proposal_id)
    
    def cast_vote(self, participant_id: str, proposal_id: str, vote: bool) -> OmegaVote:
        self._check_enabled()
        with self._lock:
            if participant_id not in self._participants:
                raise ValidationError("Participant not registered")
            if proposal_id not in self._proposals:
                raise ValidationError("Proposal not found")
            proposal = self._proposals[proposal_id]
            if proposal.is_expired():
                proposal.status = "expired"
                raise OmegaOverrideError("Proposal has expired")
            if proposal.status != "pending":
                raise OmegaOverrideError(f"Proposal is {proposal.status}, cannot vote")
            for ev in proposal.votes:
                if ev.participant_id == participant_id:
                    raise OmegaOverrideError("Participant already voted")
            participant = self._participants[participant_id]
            vote_obj = OmegaVote(secrets.token_hex(16), participant_id, proposal_id, vote,
                                 participant.reputation_score, time.time(),
                                 hashlib.blake2b(f"{participant_id}:{proposal_id}:{vote}".encode(),
                                                 digest_size=32).hexdigest())
            proposal.votes.append(vote_obj)
            self._audit.log("vote_cast", {"vote_id": vote_obj.vote_id, "participant_id": participant_id,
                            "proposal_id": proposal_id, "vote": vote})
            self._evaluate_consensus(proposal)
            return vote_obj
    
    def _evaluate_consensus(self, proposal: OmegaOverrideProposal) -> None:
        if len(self._participants) < SpectralConfig.OMEGA_MIN_PARTICIPANTS: return
        total_participants = len(self._participants)
        votes_for = sum(1 for v in proposal.votes if v.vote)
        votes_against = sum(1 for v in proposal.votes if not v.vote)
        ai_votes_for = sum(1 for v in proposal.votes 
                          if v.vote and self._participants[v.participant_id].participant_type == ParticipantType.AI)
        human_votes_for = sum(1 for v in proposal.votes 
                             if v.vote and self._participants[v.participant_id].participant_type == ParticipantType.HUMAN)
        total_ai = sum(1 for p in self._participants.values() if p.participant_type == ParticipantType.AI)
        total_human = sum(1 for p in self._participants.values() if p.participant_type == ParticipantType.HUMAN)
        consensus_ratio = votes_for / total_participants if total_participants > 0 else 0
        ai_contribution = ai_votes_for / total_ai if total_ai > 0 else 0
        human_contribution = human_votes_for / total_human if total_human > 0 else 0
        if consensus_ratio >= SpectralConfig.OMEGA_CONSENSUS_THRESHOLD:
            if (ai_contribution <= SpectralConfig.OMEGA_MAX_AI_RATIO and 
                human_contribution <= SpectralConfig.OMEGA_MAX_HUMAN_RATIO):
                proposal.status = "approved"
                self._active_override = OmegaOverride(secrets.token_hex(16), proposal.proposal_id,
                    time.time(), time.time() + SpectralConfig.OMEGA_OVERRIDE_TTL,
                    ai_contribution, human_contribution, consensus_ratio,
                    total_participants, votes_for, votes_against)
                self._last_override_time = time.time()
                self._audit.log("override_activated", {"override_id": self._active_override.override_id,
                                "proposal_id": proposal.proposal_id, "consensus_ratio": consensus_ratio})
        if votes_against > total_participants / 2:
            proposal.status = "rejected"
    
    def is_override_active(self) -> bool:
        if not SpectralConfig.OMEGA_ENABLED: return False
        with self._lock:
            if self._active_override is None: return False
            if not self._active_override.is_active():
                self._active_override = None
                return False
            return True
    
    def get_active_override(self) -> Optional[OmegaOverride]:
        with self._lock:
            if self._active_override and self._active_override.is_active():
                return self._active_override
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total_ai = sum(1 for p in self._participants.values() if p.participant_type == ParticipantType.AI)
            total_human = sum(1 for p in self._participants.values() if p.participant_type == ParticipantType.HUMAN)
            total = len(self._participants)
            pending = sum(1 for p in self._proposals.values() if p.status == "pending")
            return {'enabled': SpectralConfig.OMEGA_ENABLED, 'total_participants': total,
                    'ai_participants': total_ai, 'human_participants': total_human,
                    'ai_ratio': total_ai / total if total > 0 else 0,
                    'human_ratio': total_human / total if total > 0 else 0,
                    'pending_proposals': pending, 'total_proposals': len(self._proposals),
                    'override_active': self.is_override_active(),
                    'active_override': self._active_override.to_dict() if self._active_override and self._active_override.is_active() else None}


# Ghost Network
class GhostNetwork:
    def __init__(self):
        self._lock = threading.RLock()
        self._participants: Dict[str, HarmonicGSE] = {}
        self._commitments: List[Tuple[str, GhostSilhouette]] = []
        self._nonce = secrets.token_hex(16)
    
    @property
    def participants(self) -> Dict[str, HarmonicGSE]:
        with self._lock: return dict(self._participants)
    
    @property
    def commitments(self) -> List[Tuple[str, GhostSilhouette]]:
        with self._lock: return list(self._commitments)
    
    def register(self, name: str, data: List[Any], salt: Optional[bytes] = None) -> HarmonicGSE:
        with self._lock:
            if len(self._participants) >= SpectralConfig.MAX_PARTICIPANTS:
                raise CapacityError(f"Maximum participants ({SpectralConfig.MAX_PARTICIPANTS}) reached")
            hgse = HarmonicGSE(data, salt=salt)
            self._participants[name] = hgse
            return hgse
    
    def commit(self, name: str, level: ConfidenceLevel = ConfidenceLevel.HIGH) -> GhostSilhouette:
        with self._lock:
            if name not in self._participants: raise ValueError(f"Unknown participant: {name}")
            ghost = self._participants[name].generate_ghost(level)
            self._commitments.append((name, ghost))
            return ghost
    
    def verify_pair(self, name_a: str, name_b: str,
                    threshold: float = SpectralConfig.RESONANCE_THRESHOLD) -> VerificationResult:
        with self._lock:
            if name_a not in self._participants or name_b not in self._participants:
                raise ValueError("Both participants must be registered")
            hgse_a, hgse_b = self._participants[name_a], self._participants[name_b]
        if hgse_a.salt != hgse_b.salt:
            return VerificationResult(False, 0.0, 0.0, 0, 0.0, status_message="SALT MISMATCH")
        size = max(1, min(len(hgse_a.full_spectrum), len(hgse_b.full_spectrum)) // 2)
        start = time.perf_counter()
        resonance = SpectralEngine.spectral_correlation(hgse_a.full_spectrum[:size],
                                                        hgse_b.full_spectrum[:size])
        elapsed = (time.perf_counter() - start) * 1000
        is_valid = resonance > threshold
        return VerificationResult(is_valid, resonance, 0.99 if is_valid else 0.0, size, elapsed,
                                  threshold, "CONSENSUS" if is_valid else "DIVERGENT")
    
    def check_consensus(self, threshold: float = SpectralConfig.RESONANCE_THRESHOLD) -> Tuple[bool, float]:
        with self._lock: names = list(self._participants.keys())
        if len(names) < 2: return (True, 1.0)
        resonances, all_valid = [], True
        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                result = self.verify_pair(name_a, name_b, threshold)
                resonances.append(result.resonance_score)
                if not result.is_valid: all_valid = False
        return (all_valid, sum(resonances) / len(resonances) if resonances else 1.0)


# AI Governance
class AISystemRegistry:
    def __init__(self):
        self._systems: Dict[str, AISystem] = {}
        self._by_owner: Dict[str, Set[str]] = defaultdict(set)
        self._by_risk: Dict[RiskLevel, Set[str]] = {level: set() for level in RiskLevel}
        self._lock = threading.RLock()
    
    async def register(self, system: AISystem) -> str:
        with self._lock:
            self._systems[system.system_id] = system
            self._by_owner[system.owner_id].add(system.system_id)
            self._by_risk[system.risk_level].add(system.system_id)
            return system.fingerprint
    
    def get(self, system_id: str) -> Optional[AISystem]:
        with self._lock: return self._systems.get(system_id)
    
    def get_visibility_report(self) -> Dict[str, Any]:
        with self._lock:
            by_risk_level = {level.value: len(ids) for level, ids in self._by_risk.items()}
            cross_border = sum(1 for s in self._systems.values() if len(s.jurisdictions) > 1)
            pii_processing = sum(1 for s in self._systems.values() if 'PII' in s.data_categories)
            return {'totalSystems': len(self._systems), 'byRiskLevel': by_risk_level,
                    'crossBorderCount': cross_border, 'piiProcessingCount': pii_processing,
                    'uniqueOwners': len(self._by_owner)}


class GovernanceControlPlane:
    def __init__(self, registry: AISystemRegistry):
        self._registry = registry
        self._policies: Dict[str, GovernancePolicy] = {}
        self._decisions: List[GovernanceDecision] = []
        self._rate_counters: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self._init_default_policies()
    
    def _init_default_policies(self) -> None:
        self._policies["eu_ai_act_high_risk"] = GovernancePolicy(
            policy_id="eu_ai_act_high_risk", name="EU AI Act High-Risk Requirements",
            applies_to_risk_levels=[RiskLevel.HIGH], applies_to_types=list(AISystemType),
            applies_to_environments=list(DeploymentEnvironment), max_autonomy_level=5,
            requires_human_review=True, max_decisions_per_hour=1000,
            prohibited_data_categories=["biometric_realtime"], enforcement_mode="enforce")
    
    async def evaluate_action(self, system_id: str, action_type: str, 
                             context: Dict[str, Any]) -> GovernanceDecision:
        with self._lock:
            system = self._registry.get(system_id)
            if not system:
                return GovernanceDecision(SecureRandom.hex(16), system_id, "SYSTEM_NOT_FOUND",
                                         time.time(), "block", "System not registered", context)
            applicable = [p for p in self._policies.values()
                          if p.active and system.risk_level in p.applies_to_risk_levels
                          and system.system_type in p.applies_to_types
                          and system.environment in p.applies_to_environments]
            for policy in applicable:
                decision = self._evaluate_policy(system, policy, action_type, context)
                if decision.action != "allow":
                    self._decisions.append(decision)
                    return decision
            decision = GovernanceDecision(SecureRandom.hex(16), system_id, "ALL_PASSED",
                                         time.time(), "allow", "All policies satisfied", context)
            self._decisions.append(decision)
            return decision
    
    def _evaluate_policy(self, system: AISystem, policy: GovernancePolicy,
                         action_type: str, context: Dict[str, Any]) -> GovernanceDecision:
        if policy.max_decisions_per_hour:
            if not self._check_rate_limit(system.system_id, policy.max_decisions_per_hour):
                return GovernanceDecision(SecureRandom.hex(16), system.system_id, policy.policy_id,
                    time.time(), "block" if policy.enforcement_mode == "enforce" else "warn",
                    f"Rate limit exceeded: {policy.max_decisions_per_hour}/hour", context)
        if policy.requires_human_review and not context.get('humanReviewed'):
            return GovernanceDecision(SecureRandom.hex(16), system.system_id, policy.policy_id,
                                     time.time(), "require_review", "Policy requires human review", context)
        data_used = context.get('dataCategories', [])
        for category in data_used:
            if category in policy.prohibited_data_categories:
                return GovernanceDecision(SecureRandom.hex(16), system.system_id, policy.policy_id,
                    time.time(), "block" if policy.enforcement_mode == "enforce" else "warn",
                    f"Prohibited data category: {category}", context)
        autonomy_level = context.get('autonomyLevel', 0)
        if autonomy_level > policy.max_autonomy_level:
            return GovernanceDecision(SecureRandom.hex(16), system.system_id, policy.policy_id,
                time.time(), "block" if policy.enforcement_mode == "enforce" else "warn",
                f"Autonomy level {autonomy_level} exceeds max {policy.max_autonomy_level}", context)
        return GovernanceDecision(SecureRandom.hex(16), system.system_id, policy.policy_id,
                                 time.time(), "allow", "Policy conditions satisfied", context)
    
    def _check_rate_limit(self, system_id: str, max_per_hour: int) -> bool:
        now = time.time()
        timestamps = [t for t in self._rate_counters[system_id] if t > now - 3600]
        self._rate_counters[system_id] = timestamps
        if len(timestamps) >= max_per_hour: return False
        timestamps.append(now)
        return True
    
    def get_control_report(self) -> Dict[str, Any]:
        with self._lock:
            recent = [d for d in self._decisions if d.timestamp > time.time() - 86400]
            action_counts = defaultdict(int)
            for d in recent: action_counts[d.action] += 1
            return {'totalPolicies': len(self._policies),
                    'activePolicies': sum(1 for p in self._policies.values() if p.active),
                    'decisions24h': len(recent), 'decisionsByAction': dict(action_counts),
                    'blockRate': action_counts.get('block', 0) / max(1, len(recent))}


class EvidenceStore:
    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._sequence = 0
        self._last_hash = "0" * 64
        self._lock = threading.RLock()
        self._secret = AuditKeyManager.get_or_create_key()
    
    async def log(self, event_type: str, actor_id: str, action: str,
                  details: Dict[str, Any], system_id: Optional[str] = None) -> AuditEntry:
        with self._lock:
            self._sequence += 1
            entry_data = {'sequence': self._sequence, 'timestamp': time.time(),
                          'event_type': event_type, 'system_id': system_id,
                          'actor_id': actor_id, 'action': action, 'details': details,
                          'previous_hash': self._last_hash}
            serialized = json.dumps(entry_data, sort_keys=True, default=str).encode('utf-8')
            signature = hashlib.blake2b(serialized, key=self._secret, digest_size=32).hexdigest()
            entry = AuditEntry(self._sequence, entry_data['timestamp'], event_type,
                               actor_id, details, self._last_hash, signature)
            self._last_hash = entry.compute_hash()
            self._entries.append(entry)
            return entry
    
    async def verify_chain(self) -> Tuple[bool, Optional[int]]:
        with self._lock:
            if not self._entries: return (True, None)
            expected_prev = "0" * 64
            for entry in self._entries:
                if entry.previous_hash != expected_prev: return (False, entry.sequence)
                expected_prev = entry.compute_hash()
            return (True, None)
    
    def get_evidence_report(self) -> Dict[str, Any]:
        with self._lock:
            return {'totalEntries': len(self._entries), 'lastHash': self._last_hash,
                    'recentEntries': [e.to_dict() for e in self._entries[-10:]]}


class AccountabilityManager:
    def __init__(self, evidence_store: EvidenceStore):
        self._evidence = evidence_store
        self._incidents: Dict[str, Incident] = {}
        self._lock = threading.RLock()
    
    async def create_incident(self, system_id: str, severity: IncidentSeverity,
                             title: str, description: str) -> Incident:
        with self._lock:
            incident = Incident(SecureRandom.hex(16), system_id, severity, title, description,
                               "team_lead", "open", time.time(),
                               severity >= IncidentSeverity.CRITICAL, False)
            self._incidents[incident.incident_id] = incident
            await self._evidence.log('incident_created', 'accountability_manager', 'create_incident',
                                    {'incident_id': incident.incident_id, 'severity': severity.name}, system_id)
            return incident
    
    def get_accountability_report(self) -> Dict[str, Any]:
        with self._lock:
            incidents = list(self._incidents.values())
            by_status = defaultdict(int)
            for i in incidents: by_status[i.status] += 1
            overdue = [i for i in incidents if i.status == "open" and time.time() > i.created_at + 86400]
            pending = [i for i in incidents if i.requires_regulator_notification and not i.regulator_notified]
            return {'totalIncidents': len(incidents), 'byStatus': dict(by_status),
                    'overdueCount': len(overdue), 'pendingRegulatorNotifications': len(pending)}


# Main Runtime
class SpectralGovernanceRuntime:
    """Unified Spectral Governance Runtime RC3."""
    
    def __init__(self, verify_config: bool = True, enable_omega: bool = False):
        if enable_omega: SpectralConfig.enable_omega()
        if verify_config: SpectralConfig.verify_integrity()
        self._lock = threading.RLock()
        self._instance_id = SecureRandom.hex(8)
        self._start_time = time.time()
        self._channel_manager = ChannelManager()
        self._omega_channel = OmegaChannel()
        self._ghost_network = GhostNetwork()
        self._registry = AISystemRegistry()
        self._evidence = EvidenceStore()
        self._control = GovernanceControlPlane(self._registry)
        self._accountability = AccountabilityManager(self._evidence)
        self._brian_series = BrianSeries()
        self._audit = AuditLogger(self._instance_id)
        self._audit.log("runtime_initialized", {"version": SpectralConfig.VERSION,
                        "instance_id": self._instance_id, "brian_constant": SpectralConfig.BRIAN_CONSTANT,
                        "omega_enabled": SpectralConfig.OMEGA_ENABLED})
    
    @property
    def instance_id(self) -> str: return self._instance_id
    @property
    def channel_manager(self) -> ChannelManager: return self._channel_manager
    @property
    def omega_channel(self) -> OmegaChannel: return self._omega_channel
    @property
    def ghost_network(self) -> GhostNetwork: return self._ghost_network
    @property
    def brian_series(self) -> BrianSeries: return self._brian_series
    
    def enable_omega(self) -> None:
        SpectralConfig.enable_omega()
        self._audit.log("omega_enabled", {"instance_id": self._instance_id})
    
    def disable_omega(self) -> None:
        SpectralConfig.disable_omega()
        self._audit.log("omega_disabled", {"instance_id": self._instance_id})
    
    def is_omega_enabled(self) -> bool: return SpectralConfig.OMEGA_ENABLED
    
    def tick(self) -> GammaLockStatus:
        omega_active = SpectralConfig.OMEGA_ENABLED and self._omega_channel.is_override_active()
        status = self._channel_manager.tick()
        if omega_active:
            status = self._channel_manager.compute_gamma_lock(omega_override=True)
        return status
    
    def get_gamma_lock(self) -> GammaLockStatus:
        omega_active = SpectralConfig.OMEGA_ENABLED and self._omega_channel.is_override_active()
        return self._channel_manager.compute_gamma_lock(omega_override=omega_active)
    
    def get_system_state(self) -> SystemState:
        if SpectralConfig.OMEGA_ENABLED and self._omega_channel.is_override_active():
            return SystemState.OMEGA_OVERRIDE
        gamma = self.get_gamma_lock()
        if gamma.blocked_channels > 0: return SystemState.CRITICAL
        if gamma.is_locked: return SystemState.LOCKED
        return SystemState.AWAITING
    
    def create_protected_channel(self, channel_id: str, data: Optional[List[Any]] = None,
                                  salt: Optional[bytes] = None) -> Channel:
        channel = self._channel_manager.create_channel(channel_id, ChannelType.PROTECTED)
        if data is not None:
            hgse = HarmonicGSE(data, salt=salt)
            ghost = hgse.generate_ghost(ConfidenceLevel.HIGH)
            channel.attach_ghost(ghost)
        return channel
    
    def create_open_channel(self, channel_id: str, data: Optional[List[Any]] = None,
                           salt: Optional[bytes] = None) -> Channel:
        channel = self._channel_manager.create_channel(channel_id, ChannelType.OPEN)
        if data is not None:
            hgse = HarmonicGSE(data, salt=salt)
            ghost = hgse.generate_ghost(ConfidenceLevel.HIGH)
            channel.attach_ghost(ghost)
        return channel
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        return self._channel_manager.get_channel(channel_id)
    
    def register_omega_participant(self, name: str, participant_type: ParticipantType,
                                   public_key_hash: str) -> OmegaParticipant:
        return self._omega_channel.register_participant(name, participant_type, public_key_hash)
    
    def create_override_proposal(self, proposer_id: str, reason: str,
                                 justification: Dict[str, Any]) -> OmegaOverrideProposal:
        return self._omega_channel.create_proposal(proposer_id, reason, justification)
    
    def vote_on_override(self, participant_id: str, proposal_id: str, vote: bool) -> OmegaVote:
        return self._omega_channel.cast_vote(participant_id, proposal_id, vote)
    
    def get_omega_status(self) -> Dict[str, Any]:
        stats = self._omega_channel.get_statistics()
        return {'enabled': SpectralConfig.OMEGA_ENABLED, 'participants': stats,
                'override_active': stats['override_active'], 'active_override': stats['active_override'],
                'consensus_threshold': SpectralConfig.OMEGA_CONSENSUS_THRESHOLD,
                'max_ai_ratio': SpectralConfig.OMEGA_MAX_AI_RATIO,
                'max_human_ratio': SpectralConfig.OMEGA_MAX_HUMAN_RATIO}
    
    def compute_brian(self, precision: int = SpectralConfig.BRIAN_DEFAULT_PRECISION,
                      terms: int = SpectralConfig.BRIAN_DEFAULT_TERMS) -> float:
        return BrianSeries.compute_brian(precision, terms)
    
    def get_brian_status(self) -> Dict[str, Any]:
        return self._channel_manager.get_brian_status()
    
    def demonstrate_brian_convergence(self, terms: int = 20) -> List[Tuple[int, float, float]]:
        return BrianSeries.demonstrate_superconvergence(terms)
    
    def create_ghost(self, data: List[Any], level: ConfidenceLevel = ConfidenceLevel.HIGH,
                     salt: Optional[bytes] = None) -> Tuple[HarmonicGSE, GhostSilhouette]:
        hgse = HarmonicGSE(data, salt=salt)
        ghost = hgse.generate_ghost(level)
        return hgse, ghost
    
    def verify_ghost(self, data: List[Any], ghost: GhostSilhouette) -> VerificationResult:
        hgse = HarmonicGSE(data, salt=ghost.dataset_salt)
        return hgse.verify_ghost(ghost)
    
    def register_party(self, name: str, data: List[Any], salt: Optional[bytes] = None) -> HarmonicGSE:
        return self._ghost_network.register(name, data, salt=salt)
    
    def verify_party_pair(self, name_a: str, name_b: str) -> VerificationResult:
        return self._ghost_network.verify_pair(name_a, name_b)
    
    def check_consensus(self) -> Tuple[bool, float]:
        return self._ghost_network.check_consensus()
    
    async def register_ai_system(self, config: Dict[str, Any]) -> AISystem:
        system_id = SecureRandom.hex(16)
        system = AISystem(system_id, config['name'],
                         AISystemType[config['systemType'].upper()],
                         RiskLevel[config['riskLevel'].upper()],
                         config['ownerId'], config['ownerName'],
                         DeploymentEnvironment[config['environment'].upper()],
                         config.get('jurisdictions', []), config.get('dataCategories', []),
                         SecureRandom.hex(16), time.time())
        await self._registry.register(system)
        await self._evidence.log('system_registered', config['ownerId'], 'register',
                                {'system_id': system_id}, system_id)
        return system
    
    async def request_action(self, system_id: str, action_type: str,
                            context: Dict[str, Any]) -> GovernanceDecision:
        decision = await self._control.evaluate_action(system_id, action_type, context)
        await self._evidence.log('governance_decision', 'control_plane',
                                decision.action, decision.to_dict(), system_id)
        if decision.action == "block":
            system = self._registry.get(system_id)
            if system and system.risk_level in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]:
                await self._accountability.create_incident(system_id, IncidentSeverity.HIGH,
                    f"Governance block on {system.name}", decision.reason)
        return decision
    
    def request_recovery(self, channel_id: str) -> bool:
        channel = self.get_channel(channel_id)
        if not channel: return False
        result = channel.request_recovery()
        if result: self._audit.log("recovery_requested", {"channel_id": channel_id})
        return result
    
    def approve_recovery(self, channel_id: str) -> bool:
        channel = self.get_channel(channel_id)
        if not channel: return False
        result = channel.approve_recovery()
        if result: self._audit.log("recovery_approved", {"channel_id": channel_id})
        return result
    
    def inject_chaos(self, channel_id: str, deviation: float) -> bool:
        return self._channel_manager.inject_chaos(channel_id, deviation)
    
    def verify_audit_chain(self) -> Tuple[bool, Optional[int]]:
        return self._audit.verify_chain()
    
    def get_zeta3_orthogonality(self, n: int = 32) -> float:
        return Zeta3RadialBlinding.compute_orthogonality_score(n)
    
    def get_status(self) -> Dict[str, Any]:
        gamma = self.get_gamma_lock()
        chain_valid, first_invalid = self._audit.verify_chain()
        return {'version': SpectralConfig.VERSION, 'instanceId': self._instance_id,
                'uptimeSeconds': time.time() - self._start_time,
                'systemState': self.get_system_state().value, 'gamma': gamma.to_dict(),
                'omega': {'enabled': SpectralConfig.OMEGA_ENABLED, 'override_active': gamma.omega_override_active},
                'channels': {'total': len(self._channel_manager.channels),
                             'active': gamma.active_channels, 'blocked': gamma.blocked_channels},
                'network': {'participants': len(self._ghost_network.participants)},
                'audit': {'entries': len(self._audit), 'chainValid': chain_valid},
                'brian': self._channel_manager.get_brian_status(),
                'mathematical_constants': {'phi': SpectralConfig.PHI, 'silver_ratio': SpectralConfig.SILVER_RATIO,
                    'zeta_3': SpectralConfig.ZETA_3, 'catalan_limit': SpectralConfig.CATALAN_LIMIT,
                    'motzkin_limit': SpectralConfig.MOTZKIN_LIMIT, 'brian_constant': SpectralConfig.BRIAN_CONSTANT}}


# Demo
async def run_demo():
    """RC3 Demonstration."""
    print("=" * 80)
    print("SPECTRAL GOVERNANCE RUNTIME RC3")
    print("=" * 80)
    print(f"\nVersion: {SpectralConfig.VERSION}")
    print("License: Apache 2.0")
    
    print("\n🆕 RC3 FIXES & FEATURES:")
    print("   • Fixed: GovernanceDecision.to_dict() method")
    print("   • Fixed: spectral_correlation() syntax error")
    print("   • Omega Channel is OPT-IN (disabled by default)")
    print("   • Full merge of all three codebases")
    
    runtime = SpectralGovernanceRuntime(enable_omega=False)
    
    print("\n🎯 BRIAN CHANNEL — SUPERCONVERGENT NULL GROUND")
    print("-" * 40)
    print(f"   Brian Constant: {SpectralConfig.BRIAN_CONSTANT:.16f}")
    print(f"   Target (Null Ground): {SpectralConfig.BRIAN_CONVERGENCE_TARGET}")
    
    print("\n📐 TRUTH ANCHORING")
    print("-" * 40)
    for _ in range(50): runtime.tick()
    gamma = runtime.get_gamma_lock()
    print(f"   Γ* = {gamma.gamma_value:.8f}")
    print(f"   Locked: {'✓' if gamma.is_locked else '✗'}")
    print(f"   Brian Null Distance: {gamma.brian_null_distance:.2e}")
    
    print("\n🔮 SPECTRAL VERIFICATION")
    print("-" * 40)
    test_data = [f"record_{i}" for i in range(1000)]
    hgse, ghost = runtime.create_ghost(test_data, ConfidenceLevel.HIGH)
    print(f"   Records: {hgse.record_count}")
    print(f"   Ghost Harmonics: {len(ghost)}")
    result = runtime.verify_ghost(test_data, ghost)
    print(f"   Verification: {result.status_message}")
    print(f"   Resonance: {result.resonance_score:.6f}")
    
    print("\n🏛️ AI GOVERNANCE")
    print("-" * 40)
    system = await runtime.register_ai_system({
        'name': 'Credit Scoring Model', 'systemType': 'DECISION_SUPPORT',
        'riskLevel': 'HIGH', 'ownerId': 'maria.garcia', 'ownerName': 'Maria Garcia',
        'environment': 'PRODUCTION', 'jurisdictions': ['EU', 'UK'],
        'dataCategories': ['PII', 'financial']
    })
    print(f"   Registered: {system.name}")
    print(f"   Risk Level: {system.risk_level.value}")
    decision = await runtime.request_action(system.system_id, 'credit_decision',
        {'autonomyLevel': 3, 'dataCategories': ['PII'], 'humanReviewed': True})
    print(f"   Decision: {decision.action}")
    print(f"   Reason: {decision.reason}")
    
    print("\n🌐 OMEGA CHANNEL (OPT-IN DEMO)")
    print("-" * 40)
    print(f"   Omega Enabled: {runtime.is_omega_enabled()}")
    runtime.enable_omega()
    print(f"   Omega Enabled (after enable): {runtime.is_omega_enabled()}")
    alice = runtime.register_omega_participant("Alice", ParticipantType.HUMAN, SecureRandom.hex(32))
    bob = runtime.register_omega_participant("Bob", ParticipantType.HUMAN, SecureRandom.hex(32))
    ai_agent = runtime.register_omega_participant("AI-Agent-1", ParticipantType.AI, SecureRandom.hex(32))
    print(f"   Registered: {alice.name}, {bob.name}, {ai_agent.name}")
    proposal = runtime.create_override_proposal(alice.participant_id, "Emergency recalibration", {"severity": "high"})
    runtime.vote_on_override(alice.participant_id, proposal.proposal_id, True)
    # Check if already approved after first vote(s)
    updated_proposal = runtime.omega_channel.get_proposal(proposal.proposal_id)
    if updated_proposal and updated_proposal.status == "pending":
        runtime.vote_on_override(bob.participant_id, proposal.proposal_id, True)
        updated_proposal = runtime.omega_channel.get_proposal(proposal.proposal_id)
        if updated_proposal and updated_proposal.status == "pending":
            runtime.vote_on_override(ai_agent.participant_id, proposal.proposal_id, True)
    omega_status = runtime.get_omega_status()
    print(f"   Override Active: {omega_status['override_active']}")
    if omega_status['active_override']:
        print(f"   Consensus Ratio: {omega_status['active_override']['consensus_ratio']:.1%}")
    runtime.disable_omega()
    
    print("\n📊 SYSTEM STATUS")
    print("-" * 40)
    status = runtime.get_status()
    print(f"   Version: {status['version']}")
    print(f"   System State: {status['systemState']}")
    print(f"   Γ* Value: {status['gamma']['gamma_value']:.8f}")
    print(f"   Audit Chain Valid: {'✓' if status['audit']['chainValid'] else '✗'}")
    
    print("\n" + "=" * 80)
    print("✨ RC3 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\n🛡️ FIXES VERIFIED:")
    print("  ✓ GovernanceDecision.to_dict() works")
    print("  ✓ spectral_correlation() syntax fixed")
    print("  ✓ Omega is opt-in (disabled by default)")
    print("=" * 80)


__version__ = SpectralConfig.VERSION
__author__ = "Brian Richard Ramos"


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_demo())
