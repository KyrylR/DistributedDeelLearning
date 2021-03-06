/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package ua.univ.distributeddeellearning.COVIDClassification.Utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Class for parsing command line arguments
 */
public class JCommanderUtils {
    private static final Logger log = LoggerFactory.getLogger(JCommanderUtils.class);
    private JCommanderUtils() {
    }
    public static void parseArgs(Object obj, String[] args) {
        JCommander jcmdr = new JCommander(obj);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();  //User provides invalid input -> print the usage info
            try {
                Thread.sleep(500);
            } catch (InterruptedException exception) {
                log.error(exception.getMessage());
            }
            throw e;
        }
    }
}
