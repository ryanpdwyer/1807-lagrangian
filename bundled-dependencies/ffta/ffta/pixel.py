"""pixel.py: Contains pixel class."""
# pylint: disable=E1101,R0902,C0103
__author__ = "Durmus U. Karatay"
__copyright__ = "Copyright 2014, Ginger Lab"
__maintainer__ = "Durmus U. Karatay"
__email__ = "ukaratay@uw.edu"
__status__ = "Development"

import logging
import numpy as np
from scipy import signal as sps
from scipy import optimize as spo
from scipy import interpolate as spi

from ffta.utils import noise
from ffta.utils import cwavelet
from ffta.utils import parab

from numba import autojit


class Pixel(object):
    """
    Signal Processing to Extract Time-to-First-Peak.

    Extracts Time-to-First-Peak (tFP) from digitized Fast-Free Time-Resolved
    Electrostatic Force Microscopy (FF-trEFM) signals [1]_. It includes two
    types of frequency analysis: Hilbert Transform and Wavelet Transform.

    Parameters
    ----------
    signal_array : (n_points, n_signals) array_like
        2D real-valued signal array, corresponds to a pixel.
    params : dict
        Includes parameters for processing. The list of parameters is:

        trigger = float (in seconds)
        total_time = float (in seconds)
        sampling_rate = int (in Hz)
        drive_freq = float (in Hz)

        roi = float (in seconds)
        window = string (see documentation of scipy.signal.get_window)
        bandpass_filter = int (0: no filtering, 1: FIR filter, 2: IIR filter)
        filter_bandwidth = float (default: 5kHz)
        n_taps = integer (default: 1799)
        wavelet_analysis = bool (0: Hilbert method, 1: Wavelet Method)
        wavelet_parameter = int (default: 5)
        recombination = bool (0: FF-trEFMm, 1: Recombination)

    Attributes
    ----------
    n_points : int
        Number of points in a signal.
    n_signals : int
        Number of signals to be averaged in a pixel.
    signal_array : (n_points, n_signals) array_like
        Array that contains original signals.
    signal : (n_points,) array_like
        Signal after phase-locking and averaging.
    tidx : int
        Index of trigger in time-domain.
    phase : (n_points,) array_like
        Phase of the signal, only calculated with Hilbert Transform method.
    cwt_matrix : (n_widths, n_points) array_like
        Wavelet matrix for continuous wavelet transform.
    inst_freq : (n_points,) array_like
        Instantenous frequency of the signal.
    tfp : float
        Time from trigger to first-peak, in seconds.
    shift : float
        Frequency shift from trigger to first-peak, in Hz.

    Methods
    -------
    analyze()
        Analyzes signals and returns tfp, shift and inst_freq.

    See Also
    --------
    line: Line processing for FF-trEFM data.
    simulate: Simulation for synthetic FF-trEFM data.
    scipy.signal.get_window: Windows for signal processing.

    Notes
    -----
    Frequency shift from wavelet analysis is not in Hertz. It should be used
    with caution.

    analyze() does not raise an exception if there is one, however it logs the
    exception if logging is turned on. This is implemented this way to avoid
    crashing in the case of exception when this method is called from C API.

    References
    ----------
    .. [1] Giridharagopal R, Rayermann GE, Shao G, et al. Submicrosecond time
       resolution atomic force microscopy for probing nanoscale dynamics.
       Nano Lett. 2012;12(2):893-8.

    Examples
    --------
    >>> from ffta import pixel, utils
    >>>
    >>> signal_file = '../data/SW_0000.ibw'
    >>> params_file = '../data/parameters.cfg'
    >>>
    >>> signal_array = utils.load.signal(signal_file)
    >>> n_pixels, params = utils.load.configuration(params_file)
    >>>
    >>> p = pixel.Pixel(signal_array, params)
    >>> tfp, shift, inst_freq = p.analyze()

    """

    def __init__(self, signal_array, params):

        # Create parameter attributes for optional parameters.
        # They will be overwritten by following for loop if they exist.
        self.n_taps = 1799
        self.filter_bandwidth = 5000
        self.wavelet_analysis = False
        self.wavelet_parameter = 5
        self.recombination = False

        # Read parameter attributes from parameters dictionary.
        for key, value in params.items():

            setattr(self, key, value)

        # Assign values from inputs.
        self.signal_array = signal_array
        self.tidx = int(self.trigger * self.sampling_rate)
        self.n_points, self.n_signals = signal_array.shape

        # Keep the original values for restoring the signal properties.
        self._tidx_orig = self.tidx
        self._n_points_orig = signal_array.shape[0]

        # Initialize attributes that are going to be assigned later.
        self.signal = None
        self.phase = None
        self.inst_freq = None
        self.tfp = None
        self.shift = None
        self.cwt_matrix = None

        return

    def remove_dc(self):
        """Removes DC components from signals."""

        self.signal_array -= self.signal_array.mean(axis=0)

        return

    def phase_lock(self):
        """Phase-locks signals in the signal array. This also cuts signals."""

        # Phase-lock signals.
        self.signal_array, self.tidx = noise.phase_lock(
            self.signal_array, self.tidx,
            np.ceil(self.sampling_rate / self.drive_freq))

        # Update number of points after phase-locking.
        self.n_points = self.signal_array.shape[0]

        return

    def average(self):
        """Averages signals."""

        self.signal = self.signal_array.mean(axis=1)

        return

    def check_drive_freq(self):
        """Calculates drive frequency of averaged signals, and check against
           the given drive frequency."""

        n_fft = 2 ** int(np.log2(self.tidx))  # For FFT, power of 2.
        dfreq = self.sampling_rate / n_fft  # Frequency separation.

        # Calculate drive frequency from maximum power of the FFT spectrum.
        signal = self.signal[:n_fft]
        fft_amplitude = np.abs(np.fft.rfft(signal))
        drive_freq = fft_amplitude.argmax() * dfreq

        # Difference between given and calculated drive frequencies.
        difference = np.abs(drive_freq - self.drive_freq)

        # If difference is too big, reassign. Otherwise, continue.
        if difference >= dfreq:

            self.drive_freq = drive_freq

        return

    def apply_window(self):
        """Applies the window given in parameters."""

        self.signal *= sps.get_window(self.window, self.n_points)

        return

    def fir_filter(self):
        """Filters signal with a FIR bandpass filter."""

        # Calculate bandpass region from given parameters.
        nyq_rate = 0.5 * self.sampling_rate
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / nyq_rate
        freq_high = (self.drive_freq + bw_half) / nyq_rate

        band = [freq_low, freq_high]

        # Create taps using window method.
        taps = sps.firwin(self.n_taps, band, pass_zero=False,
                          window='parzen')

        self.signal = sps.fftconvolve(self.signal, taps, mode='same')
        self.tidx -= (self.n_taps - 1) / 2

        return

    def iir_filter(self):
        """Filters signal with two Butterworth filters (one lowpass,
        one highpass) using filtfilt. This method has linear phase and no
        time delay. Do not use for production."""

        # Calculate bandpass region from given parameters.
        nyq_rate = 0.5 * self.sampling_rate
        bw_half = self.filter_bandwidth / 2

        freq_low = (self.drive_freq - bw_half) / nyq_rate
        freq_high = (self.drive_freq + bw_half) / nyq_rate

        # Do a high-pass filtfilt operation.
        b, a = sps.butter(5, freq_low, btype='high')
        self.signal = sps.filtfilt(b, a, self.signal)

        # Do a low-pass filtfilt operation.
        b, a = sps.butter(5, freq_high, btype='low')
        self.signal = sps.filtfilt(b, a, self.signal)

        return

    def hilbert_transform(self):
        """Gets the analytical signal doing a Hilbert transform."""

        self.signal = sps.hilbert(self.signal)

        return

    def calculate_phase(self, correct_slope=True):
        """Gets the phase of the signal and correct the slope by removing
        the drive phase."""

        # Unwrap the phase.
        self.phase = np.unwrap(np.angle(self.signal))

        if correct_slope:

            # Remove the drive from phase.
            self.phase -= (2 * np.pi * self.drive_freq *
                           np.arange(self.n_points) / self.sampling_rate)

            # A curve fit on the initial part to make sure that it worked.
            start = int(0.3 * self.tidx)
            end = int(0.7 * self.tidx)
            fit = self.phase[start:end]

            xfit = np.polyfit(np.arange(start, end), fit, 1)

            # Remove the fit from phase.
            self.phase -= (xfit[0] * np.arange(self.n_points)) + xfit[1]

        return

    def calculate_inst_freq(self):
        """Calculates the first derivative of the phase using Savitzky-Golay
        filter."""

        dtime = 1 / self.sampling_rate  # Time step.

        # Do a Savitzky-Golay smoothing derivative
        # using 5 point 1st order polynomial.
        self.inst_freq = sps.savgol_filter(self.phase, 5, 1, deriv=1,
                                           delta=dtime)

        # Bring trigger to zero.
        self.inst_freq -= self.inst_freq[self.tidx]

        return

    def find_minimum(self):
        """Finds when the minimum of instantenous frequency happens."""

        # Cut the signal into region of interest.
        ridx = int(self.roi * self.sampling_rate)
        cut = self.inst_freq[self.tidx:(self.tidx + ridx)]

        # Define a spline to be used in finding minimum.
        x = np.arange(ridx)
        y = cut
        func = spi.UnivariateSpline(x, y, k=4, ext=3)

        # Find the minimum of the spline using TNC method.
        res = spo.minimize(func, cut.argmin(),
                           method='TNC', bounds=((0, ridx),))
        idx = res.x[0]

        # Do index to time conversion and find shift.
        self.tfp = idx / self.sampling_rate
        self.shift = func(0) - func(idx)

        return

    def restore_signal(self):
        """Restores the signal length and position of trigger to original
        values."""

        # Difference between current and original values.
        d_trig = int(self._tidx_orig - self.tidx)
        d_points = int(self._n_points_orig - self.n_points)

        self.d_trig = d_trig

        # Check if the signal length can accomodate the shift or not.
        if d_trig >= d_points:

            # Pad from left and set the original length.
            self.inst_freq = np.pad(self.inst_freq, (d_trig, 0), 'edge')
            self.inst_freq = self.inst_freq[:self._n_points_orig]

        else:

            # Calculate how many points is needed for padding from right.
            pad_right = d_points - d_trig
            self.inst_freq = np.pad(self.inst_freq, (d_trig, pad_right),
                                    'edge')

        # Set the public variables back to original values.
        self.tidx = self._tidx_orig
        self.n_points = self._n_points_orig

        return

    @autojit
    def __get_cwt__(self):
        """Generates the CWT using Morlet wavelet. Returns a 2D Matrix."""

        w0 = self.wavelet_parameter
        wavelet_increment = 0.5  # Reducing this has little benefit.

        cwt_scale = ((w0 + np.sqrt(2 + w0 ** 2)) /
                     (4 * np.pi * self.drive_freq / self.sampling_rate))

        widths = np.arange(cwt_scale * 0.9, cwt_scale * 1.1,
                           wavelet_increment)

        cwt_matrix = cwavelet.cwt(self.signal, dt=1, scales=widths, p=w0)
        self.cwt_matrix = np.abs(cwt_matrix)

        return w0, wavelet_increment, cwt_scale

    def calculate_cwt_freq(self):
        """Fits a curve to each column in the CWT matrix to generate the
        frequency."""

        # Generate necessary tools for wavelet transform.
        w0, wavelet_increment, cwt_scale = self.__get_cwt__()

        _, n_points = np.shape(self.cwt_matrix)
        inst_freq = np.empty(n_points)

        for i in xrange(n_points):

            cut = self.cwt_matrix[:, i]
            inst_freq[i], _ = parab.fit(cut, np.argmax(cut))

        inst_freq = (inst_freq * wavelet_increment + 0.9 * cwt_scale)
        inst_freq = ((w0 + np.sqrt(2 + w0 ** 2)) /
                     (4 * np.pi * inst_freq[:] / self.sampling_rate))

        self.inst_freq = inst_freq - inst_freq[self.tidx]

        return

    def analyze(self):
        """
        Analyzes the pixel with the given method.

        Returns
        -------
        tfp : float
            Time from trigger to first-peak, in seconds.
        shift : float
            Frequency shift from trigger to first-peak, in Hz.
        inst_freq : (n_points,) array_like
            Instantenous frequency of the signal.

        Notes
        -----
        It does not raise an exception if there is one, however it logs
        the exception if logging is turned on. This is implemented this way to
        avoid crashing in the case of exception when this method is called
        from C API.

        """

        try:
            # Remove DC component, first.
            self.remove_dc()

            # Phase-lock signals.
            self.phase_lock()

            # Average signals.
            self.average()

            # Remove DC component again, introduced by phase-locking.
            self.remove_dc()

            # Check the drive frequency.
            self.check_drive_freq()

            if self.wavelet_analysis:

                # Calculate instantenous frequency using wavelet transform.
                self.calculate_cwt_freq()

            else:

                # Apply window.
                self.apply_window()

                # Filter the signal with a filter, if wanted.
                if self.bandpass_filter == 1:

                    self.fir_filter()

                elif self.bandpass_filter == 2:

                    self.iir_filter()

                # Get the analytical signal doing a Hilbert transform.
                self.hilbert_transform()

                # Calculate the phase from analytic signal.
                self.calculate_phase()

                # Calculate instantenous frequency.
                self.calculate_inst_freq()

            # If it's a recombination image invert it to find minimum.
            if self.recombination:

                self.inst_freq = self.inst_freq * -1

            # Find where the minimum is.
            self.find_minimum()

            # Restore the length.
            self.restore_signal()

        # If caught any exception, set everything to zero and log it.
        except Exception as exception:

            self.tfp = 0
            self.shift = 0
            self.inst_freq = np.zeros(self._n_points_orig)

            logging.exception(exception, exc_info=True)

        return self.tfp, self.shift, self.inst_freq
