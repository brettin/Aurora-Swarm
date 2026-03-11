#!/usr/bin/env perl

use strict;
use warnings;
use List::Util qw(min max sum);

# Read data from STDIN
my @seconds;
while (my $line = <STDIN>) {
    chomp $line;
    next if $line =~ /^\s*$/;  # Skip empty lines
    
    # Parse line: hostname time seconds
    # Example: x4514c4s5b0n0	00:04:31	271 seconds
    my @fields = split(/\s+/, $line);
    
    if (@fields >= 3) {
        # Extract the numeric value from the seconds column (could be "271" or "271 seconds")
        my $seconds_value = $fields[2];
        $seconds_value =~ s/\s*seconds?\s*$//i;  # Remove "seconds" suffix if present
        
        if ($seconds_value =~ /^\d+(\.\d+)?$/) {
            push @seconds, $seconds_value;
        }
    }
}

# Check if we have data
if (@seconds == 0) {
    die "Error: No valid data found in input\n";
}

# Sort for median calculation
my @sorted = sort { $a <=> $b } @seconds;

# Calculate statistics
my $n = scalar @seconds;
my $sum = sum(@seconds);
my $mean = $sum / $n;

# Calculate standard deviation
my $sum_squared_diff = sum(map { ($_ - $mean) ** 2 } @seconds);
my $variance = $sum_squared_diff / $n;
my $stddev = sqrt($variance);

# Calculate sample standard deviation (n-1)
my $sample_variance = $n > 1 ? $sum_squared_diff / ($n - 1) : 0;
my $sample_stddev = sqrt($sample_variance);

# Calculate median
my $median;
if ($n % 2 == 0) {
    $median = ($sorted[$n/2 - 1] + $sorted[$n/2]) / 2;
} else {
    $median = $sorted[int($n/2)];
}

# Calculate quartiles
my $q1_idx = int($n * 0.25);
my $q3_idx = int($n * 0.75);
my $q1 = $sorted[$q1_idx];
my $q3 = $sorted[$q3_idx];

my $min = min(@seconds);
my $max = max(@seconds);
my $range = $max - $min;

# Print results in tab-delimited format (single row)
print "Sample_size\tMean\tMedian\tStd_dev_population\tStd_dev_sample\tVariance\tMinimum\tMaximum\tRange\tQ1_25th_percentile\tQ3_75th_percentile\tIQR\n";
printf "%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n", 
    $n, $mean, $median, $stddev, $sample_stddev, $variance, $min, $max, $range, $q1, $q3, ($q3 - $q1);
