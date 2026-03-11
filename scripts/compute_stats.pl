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

# Print results
print "=" x 60 . "\n";
print "STATISTICS SUMMARY\n";
print "=" x 60 . "\n";
printf "Sample size:           %d\n", $n;
printf "Mean:                  %.2f seconds\n", $mean;
printf "Median:                %.2f seconds\n", $median;
printf "Standard deviation:    %.2f seconds (population)\n", $stddev;
printf "Std dev (sample):      %.2f seconds (n-1)\n", $sample_stddev;
printf "Variance:              %.2f\n", $variance;
printf "Minimum:               %.2f seconds\n", $min;
printf "Maximum:               %.2f seconds\n", $max;
printf "Range:                 %.2f seconds\n", $range;
printf "Q1 (25th percentile):  %.2f seconds\n", $q1;
printf "Q3 (75th percentile):  %.2f seconds\n", $q3;
printf "IQR (Q3-Q1):           %.2f seconds\n", $q3 - $q1;
print "=" x 60 . "\n";

# Optional: Show distribution summary
my @bins = (0, 270, 275, 280, 285, 290, 295, 300, 1000000);
my %distribution;
foreach my $val (@seconds) {
    for (my $i = 0; $i < @bins - 1; $i++) {
        if ($val >= $bins[$i] && $val < $bins[$i+1]) {
            my $label = $bins[$i] == 0 ? sprintf("< %d", $bins[$i+1]) :
                        $bins[$i+1] == 1000000 ? sprintf(">= %d", $bins[$i]) :
                        sprintf("%d-%d", $bins[$i], $bins[$i+1]-1);
            $distribution{$label}++;
            last;
        }
    }
}

if (keys %distribution) {
    print "\nDISTRIBUTION:\n";
    foreach my $bin (sort keys %distribution) {
        my $count = $distribution{$bin};
        my $pct = ($count / $n) * 100;
        my $bar = '#' x int($pct / 2);
        printf "  %-12s: %3d (%.1f%%) %s\n", $bin, $count, $pct, $bar;
    }
}
